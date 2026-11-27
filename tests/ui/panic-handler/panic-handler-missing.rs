//@ dont-check-compiler-stderr

#![feature(lang_items)]
#![feature(panic_unwind)]
#![no_main]
#![no_std]

extern crate unwind;

//~? ERROR `#[panic_handler]` function required, but not found
