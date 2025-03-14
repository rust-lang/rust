#![no_std]
#![crate_type = "lib"]

use core::panic::PanicInfo;

#[warn(clippy::all)]
pub fn main() {
    let mut a = 42;
    let mut b = 1337;

    a = b;
    //~^ almost_swapped

    b = a;
}
