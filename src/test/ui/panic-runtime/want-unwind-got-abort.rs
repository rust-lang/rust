// build-fail
// needs-unwind
// error-pattern:is not compiled with this crate's panic strategy `unwind`
// aux-build:panic-runtime-abort.rs
// aux-build:panic-runtime-lang-items.rs
// ignore-wasm32-bare compiled with panic=abort by default

#![no_std]
#![no_main]

extern crate panic_runtime_abort;
extern crate panic_runtime_lang_items;
