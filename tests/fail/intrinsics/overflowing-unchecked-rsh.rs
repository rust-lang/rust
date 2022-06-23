#![feature(core_intrinsics)]

use std::intrinsics::*;

fn main() {
    unsafe {
        let _n = unchecked_shr(1i64, 64);
        //~^ ERROR: overflowing shift by 64 in `unchecked_shr`
    }
}
