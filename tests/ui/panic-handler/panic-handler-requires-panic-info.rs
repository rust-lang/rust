//@ compile-flags:-C panic=abort

#![feature(lang_items)]
#![feature(no_core)]
#![no_core]
#![no_main]

#[panic_handler]
//~^ ERROR cannot find attribute `panic_handler` in this scope
fn panic() -> ! {
    loop {}
}

#[lang = "sized"]
trait Sized {}
