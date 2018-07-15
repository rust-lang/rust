#![feature(intrinsics)]

mod rusti {
    extern "rust-intrinsic" {
        pub fn ctlz_nonzero<T>(x: T) -> T;
    }
}

pub fn main() {
    unsafe {
        use rusti::*;

        ctlz_nonzero(0u8); //~ ERROR constant evaluation error
        //~^ NOTE ctlz_nonzero called on 0
    }
}
