// error-pattern:is incompatible with this crate's strategy of `unwind`
// aux-build:panic-runtime-abort.rs
// aux-build:panic-runtime-lang-items.rs
// ignore-wasm32-bare compiled with panic=abort by default

#![no_std]

extern crate panic_runtime_abort;
extern crate panic_runtime_lang_items;

fn main() {}
