#![no_std]
#![feature(lang_items, start, libc)]
#![crate_type = "lib"]

use core::panic::PanicInfo;

#[warn(clippy::all)]
fn main() {
    let mut a = 42;
    let mut b = 1337;

    a = b;
    b = a;
}
