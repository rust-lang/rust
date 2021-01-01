// build-fail
// dont-check-compiler-stderr
// error-pattern:cannot link together two panic runtimes: panic_runtime_unwind and panic_runtime_unwind2
// ignore-tidy-linelength
// aux-build:panic-runtime-unwind.rs
// aux-build:panic-runtime-unwind2.rs
// aux-build:panic-runtime-lang-items.rs

#![no_std]
#![no_main]

extern crate panic_runtime_unwind;
extern crate panic_runtime_unwind2;
extern crate panic_runtime_lang_items;

fn main() {}
