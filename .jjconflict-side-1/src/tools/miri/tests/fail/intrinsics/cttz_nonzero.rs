#![feature(core_intrinsics)]

fn main() {
    unsafe {
        use std::intrinsics::*;

        cttz_nonzero(0u8); //~ ERROR: `cttz_nonzero` called on 0
    }
}
