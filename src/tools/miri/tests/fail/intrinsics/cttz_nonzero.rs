#![feature(intrinsics)]

mod rusti {
    extern "rust-intrinsic" {
        pub fn cttz_nonzero<T>(x: T) -> T;
    }
}

pub fn main() {
    unsafe {
        use crate::rusti::*;

        cttz_nonzero(0u8); //~ ERROR: `cttz_nonzero` called on 0
    }
}
