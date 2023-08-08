//@compile-flags: -Clink-arg=-nostartfiles
//@ignore-target-apple

#![feature(lang_items, start, libc)]
#![no_std]

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

#[warn(clippy::main_recursion)]
#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let x = N.load(Ordering::Relaxed);
    N.store(x + 1, Ordering::Relaxed);

    if x < 3 {
        main(_argc, _argv);
    }

    0
}

#[allow(clippy::empty_loop)]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
