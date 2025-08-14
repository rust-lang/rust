//@ compile-flags:-C panic=abort

#![feature(lang_items)]
#![feature(no_core)]
#![no_core]
#![no_main]

#[panic_handler]
fn panic() -> ! {
    //~^ ERROR requires `panic_info` lang_item
    loop {}
}

#[lang = "sized"]
trait Sized {}
