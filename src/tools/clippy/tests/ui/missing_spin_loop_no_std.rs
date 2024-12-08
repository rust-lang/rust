#![warn(clippy::missing_spin_loop)]
#![feature(lang_items, start, libc)]
#![no_std]

use core::sync::atomic::{AtomicBool, Ordering};

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    // This should trigger the lint
    let b = AtomicBool::new(true);
    // This should lint with `core::hint::spin_loop()`
    while b.load(Ordering::Acquire) {}
    0
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
