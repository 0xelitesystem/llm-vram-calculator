# LLM VRAM Calculator

Estimate the GPU memory a local LLM needs, with the arithmetic shown, an honest range instead of a fake-precise number, and a fits-or-not verdict for common GPUs.

**Live demo:** https://0xelitesystem.github.io/llm-vram-calculator/

## Use

1. Pick a model preset, or enter the parameter count and the architecture (layers, attention heads, key/value heads, head dimension) from the model's `config.json`.
2. Choose the weight quantization and KV cache precision, then set context length and batch size.
3. Pick your GPU, or type your VRAM in **My VRAM (GiB)**.
4. Read the estimate range, the fits-or-not verdict and the suggested fixes, then press **Copy summary**.

## Why this exists

A VRAM estimate that computes the KV cache from attention heads instead of key/value heads overstates grouped-query models by the group size, and one that prints a single number hides how much of it is guesswork. This calculator shows every term with its arithmetic and gives a range. It is one HTML file that runs in your browser, with no tracking and no server, under the MIT license.

## Features

- **Weights, KV cache, overhead as separate line items**, each with its derivation printed underneath so you can check the math.
- **Correct KV cache formula**, shown on the page: `2 x layers x kv_heads x head_dim x context x batch x bytes_per_element`. It uses key/value heads, not attention heads, so grouped-query and multi-query models are not overestimated by 4x or 8x.
- **Quantization presets** for FP16/BF16, FP8, INT8 and the common GGUF classes (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q3_K_M, Q2_K). Bits per weight is exact where the format fixes it, labelled approximate where the class mixes tensor types, and editable in every case.
- **Separate KV cache precision** (FP16, FP8, INT8, or your own number), because quantizing the cache is a real lever and often the cheapest one.
- **Architecture presets** for well-known model shapes (Llama 2 and 3 families, Mistral 7B, Mixtral 8x7B, Qwen2.5 sizes, Gemma 2 9B) that fill layers, attention heads, key/value heads and head dimension. Every field stays editable.
- **Fits-or-not verdict** against consumer, workstation and datacenter cards, plus an Apple unified-memory option with a note that unified memory does not behave like dedicated VRAM. Free-entry "my VRAM" field for anything not in the list.
- **Concrete fixes when it does not fit**: the highest-quality quantization that would fit, the largest context that would fit (solved, not guessed), what an 8-bit KV cache would save, and how many layers you could offload.
- **Partial offload estimate**: how many layers land on the GPU and what spills to system RAM, with a plain warning that CPU offload is dramatically slower and no fake tokens-per-second claim.
- **Honest error bars**: results are a range. Weights and KV cache are labelled near-exact arithmetic; activation, workspace and runtime overhead are labelled estimates, because that is what they are.
- Copyable plain-text summary of the whole estimate, including the caveats.
- Dark theme by default, light/dark toggle that persists, works down to a 360px screen, no external dependencies.

## How it works

Three terms, added up.

**Weights** are `parameters x bits_per_weight / 8`. Near-exact once you know the parameter count and the real bits per weight. For FP16, FP8, INT8, Q8_0 and Q4_0 the bits per weight is fixed by the format (Q8_0 is 32 int8 values plus one FP16 scale per block, so `(32 x 8 + 16) / 32 = 8.5`). For the mixed GGUF classes ending in `_K_M` or `_K_S` the whole-file average is approximate and labelled as such, because those classes keep some tensors at higher precision. The spread is real: Q4_K_M averages about 4.83 bits on LLaMA-7B and about 4.90 on Llama 3.1 8B, and Q2_K moves from about 3.35 to about 3.16 across the same two. If you already have the file, `file_size_in_bytes x 8 / parameter_count` gives you the exact number, and the field is editable.

Parameter counts in the presets are the safetensors tensor totals from each model repo, not the rounded figure on the model card. Where the two disagree the preset says so: Qwen2.5 32B is 32.76B of actual tensors against a card that says 32.5B.

**KV cache** is `2 x layers x kv_heads x head_dim x context x batch x bytes_per_element`. The 2 is one key plus one value. The part most calculators get wrong is `kv_heads`: grouped-query attention gives a model far fewer key/value heads than attention heads (Llama 3.1 8B has 32 attention heads and 8 key/value heads, Qwen2.5 7B has 4), so using attention heads inflates the answer by the group size. Pull these four numbers from the model's `config.json`: `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`.

**Activation, workspace and runtime overhead** is the estimated part, and the tool says so every time it shows it. It covers attention and MLP scratch buffers, the logits buffer, CUDA graphs, allocator fragmentation, and the CUDA context itself, which costs a few hundred MB per process before any weights load. The calculator applies an editable percentage band to weights plus KV, plus a flat runtime allowance. That is a rule of thumb, not a measurement, and it is the reason every result on the page is a range.

What the tool deliberately does not do: it does not predict tokens per second, it does not invent benchmark numbers, and it does not pretend the overhead figure is measured. Real usage also depends on your inference engine. llama.cpp allocates the cache you ask for up front and supports quantized K and V, though a quantized V cache requires flash attention and it will refuse to start without it. vLLM preallocates a paged pool sized by `gpu_memory_utilization`, which currently defaults to 0.92, so it will appear to fill the card regardless. transformers keeps the cache in FP16 with no paging. Flash attention removes the attention score matrix from memory, which matters most at long context and large batch.

The formula also does not describe every architecture. Multi-head latent attention (the DeepSeek V2/V3 family) compresses the cache and would be badly overestimated here. Sliding-window and interleaved local attention (Mistral's window, Gemma 2's alternating layers) cap the cache instead of growing it with full context. State-space and hybrid models carry a fixed-size state. Mixture-of-experts models follow the KV math fine, but the weight term has to count every expert, not the active ones. The page lists all of this next to the result.

Companion reference, if the question behind the question is whether to self-host at all: [local-llm-vs-api-reference](https://github.com/0xelitesystem/local-llm-vs-api-reference).

## Privacy

Everything runs in your browser. Nothing is uploaded, there is no analytics, no tracking and no network request of any kind. The page is a single HTML file with no external dependencies. Open it from disk with the network cable unplugged and it works exactly the same. The only thing stored is your light/dark preference, in localStorage on your own machine.

## Run locally

```bash
git clone https://github.com/0xelitesystem/llm-vram-calculator
cd llm-vram-calculator
```

Open `index.html` in any modern browser. Or serve the folder with `python -m http.server 8000` and visit http://localhost:8000/.

## Build

No build step. The whole tool is one `index.html` file with its CSS and JavaScript inline, and nothing to install.

## License

MIT. See [LICENSE](LICENSE).

## More

- Catalog of every tool: https://0xelitesystem.github.io/
- https://elitesystem.ai
