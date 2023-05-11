#![crate_type = "rlib"]
#![feature(lang_items)]
#![feature(panic_unwind)]
#![no_std]

extern crate panic_unwind;

#[panic_handler]
pub fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
extern "C" fn __rust_drop_panic() -> ! {
    loop {}
}

#[no_mangle]
extern "C" fn __rust_foreign_exception() -> ! {
    loop {}
}

#[lang = "eh_personality"]
fn eh_personality() {
    loop {}
}
