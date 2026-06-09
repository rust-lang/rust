#![warn(clippy::missing_spin_loop)]
#![crate_type = "lib"]
#![no_std]

use core::sync::atomic::{AtomicBool, Ordering};

pub fn main(_argc: isize, _argv: *const *const u8) -> isize {
    // This should trigger the lint
    let b = AtomicBool::new(true);
    // This should lint with `core::hint::spin_loop()`
    while b.load(Ordering::Acquire) {}
    //~^ missing_spin_loop
    0
}
