//@ build-fail
//@ needs-unwind
//@ error-pattern:is incompatible with this crate's strategy of `unwind`
//@ aux-build:panic-runtime-abort.rs
//@ aux-build:wants-panic-runtime-abort.rs
//@ aux-build:panic-runtime-lang-items.rs

#![no_std]
#![no_main]

extern crate wants_panic_runtime_abort;
extern crate panic_runtime_lang_items;
