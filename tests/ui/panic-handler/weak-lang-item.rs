//@ aux-build:weak-lang-items.rs
//@ needs-unwind since it affects the error output
//@ ignore-emscripten missing eh_catch_typeinfo lang item

#![no_std]

extern crate core; //~ ERROR the name `core` is defined multiple times
extern crate weak_lang_items;

fn main() {}

//~? ERROR `#[panic_handler]` function required, but not found
//~? ERROR unwinding panics are not supported without std
