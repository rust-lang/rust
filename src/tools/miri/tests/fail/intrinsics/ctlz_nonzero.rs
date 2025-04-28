#![feature(intrinsics)]

mod rusti {
    #[rustc_intrinsic]
    pub unsafe fn ctlz_nonzero<T>(x: T) -> u32;
}

pub fn main() {
    unsafe {
        use crate::rusti::*;

        ctlz_nonzero(0u8); //~ ERROR: `ctlz_nonzero` called on 0
    }
}
