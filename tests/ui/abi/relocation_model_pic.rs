// run-pass
// compile-flags: -C relocation-model=pic
// ignore-emscripten no pic
// ignore-wasm

#![feature(cfg_relocation_model)]

#[cfg(relocation_model = "pic")]
fn main() {}
