#![feature(core_intrinsics)]

use std::intrinsics::*;

fn main() {
    unsafe {
        let _n = unchecked_rem(3u32, 0); //~ ERROR: calculating the remainder with a divisor of zero
    }
}
