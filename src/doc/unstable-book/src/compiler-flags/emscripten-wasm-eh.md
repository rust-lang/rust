# `emscripten-wasm-eh`

Use the WebAssembly exception handling ABI to unwind for the
`wasm32-unknown-emscripten`. If compiling with this setting, the `emcc` linker
should be invoked with `-fwasm-exceptions`. If linking with C/C++ files, the
C/C++ files should also be compiled with `-fwasm-exceptions`.
