#![no_std]
#![feature(lang_items, start, libc)]
#![crate_type = "lib"]

use core::panic::PanicInfo;

#[warn(clippy::all)]
fn main() {
    let mut a = 42;
    let mut b = 1337;

    a = b;
    //~^ ERROR: this looks like you are trying to swap `a` and `b`
    //~| NOTE: or maybe you should use `core::mem::replace`?
    b = a;
}
