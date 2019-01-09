// compile-flags:-C panic=abort

#![feature(lang_items)]
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    loop {}
}

#[lang = "panic_impl"]
fn panic2(info: &PanicInfo) -> ! { //~ ERROR duplicate lang item found: `panic_impl`.
    loop {}
}
