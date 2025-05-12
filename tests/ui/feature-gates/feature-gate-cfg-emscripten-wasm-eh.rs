//@ compile-flags: --check-cfg=cfg(emscripten_wasm_eh)
#[cfg(not(emscripten_wasm_eh))]
//~^ ERROR `cfg(emscripten_wasm_eh)` is experimental
fn main() {}
