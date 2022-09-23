// aux-build:weak-lang-items.rs
// error-pattern: `#[panic_handler]` function required, but not found
// error-pattern: language item required, but not found: `eh_personality`
// needs-unwind since it affects the error output
// ignore-emscripten missing eh_catch_typeinfo lang item

#![no_std]

extern crate core;
extern crate weak_lang_items;

fn main() {}
