#![feature(core_intrinsics)]

use std::intrinsics::*;

//error-pattern: overflowing shift by 64 in `unchecked_shr`

fn main() {
    unsafe {
        let _n = unchecked_shr(1i64, 64);
    }
}
