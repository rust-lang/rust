//@aux-build:weak-lang-items.rs
//@error-in-other-file: `#[panic_handler]` function required, but not found
//@error-in-other-file: language item required, but not found: `eh_personality`
// needs-unwind since it affects the error output
//@ignore-target-emscripten missing eh_catch_typeinfo lang item

#![no_std]

extern crate core;
extern crate weak_lang_items;

fn main() {}
