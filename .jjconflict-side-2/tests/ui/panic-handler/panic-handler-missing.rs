//@ dont-check-compiler-stderr

#![feature(lang_items)]
#![no_main]
#![no_std]

#[lang = "eh_personality"]
fn eh() {}

//~? ERROR `#[panic_handler]` function required, but not found
