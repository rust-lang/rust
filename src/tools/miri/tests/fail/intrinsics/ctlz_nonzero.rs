#![feature(intrinsics)]

mod rusti {
    extern "rust-intrinsic" {
        pub fn ctlz_nonzero<T>(x: T) -> T;
    }
}

pub fn main() {
    unsafe {
        use crate::rusti::*;

        ctlz_nonzero(0u8); //~ ERROR: `ctlz_nonzero` called on 0
    }
}
