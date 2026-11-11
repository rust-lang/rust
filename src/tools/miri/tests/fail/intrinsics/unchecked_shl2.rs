#![feature(core_intrinsics)]
use std::intrinsics;

fn main() {
    unsafe {
        let _n = intrinsics::unchecked_shl(1i8, -1);
        //~^ ERROR: overflowing shift by -1 in `unchecked_shl`
    }
}
