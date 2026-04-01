//@ no-prefer-dynamic
#![crate_type = "lib"]
#![no_std]
#![feature(lang_items)]

use core::panic::PanicInfo;
use core::sync::atomic::{self, Ordering};

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        atomic::compiler_fence(Ordering::SeqCst);
    }
}

#[lang = "eh_personality"]
fn foo() {}
