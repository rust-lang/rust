// compile-flags:-C panic=abort
// error-pattern: language item required, but not found: `panic_info`

#![feature(lang_items)]
#![feature(no_core)]
#![no_core]
#![no_main]

#[panic_handler]
fn panic() -> ! {
    loop {}
}

#[lang = "sized"]
trait Sized {}
