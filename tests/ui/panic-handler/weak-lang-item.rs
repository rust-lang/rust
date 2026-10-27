//@ edition:2015
//@ aux-build:weak-lang-items.rs
//@ needs-unwind since it affects the error output
//~? ERROR: `#[panic_handler]` function required
//~? ERROR: unwinding panics are not supported without std

#![no_std]

extern crate core; //~ ERROR the name `core` is defined multiple times
extern crate weak_lang_items;

fn main() {}
