#![feature(core_intrinsics)]

use std::intrinsics::*;

fn main() {
    unsafe {
        let _n = unchecked_div(1i64, 0); //~ERROR: dividing by zero
    }
}
