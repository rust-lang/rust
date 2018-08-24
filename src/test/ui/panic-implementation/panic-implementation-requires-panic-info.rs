// compile-flags:-C panic=abort
// error-pattern: language item required, but not found: `panic_info`

#![feature(lang_items)]
#![feature(no_core)]
#![feature(panic_implementation)]
#![no_core]
#![no_main]

#[panic_implementation]
fn panic() -> ! {
    loop {}
}

#[lang = "sized"]
trait Sized {}
