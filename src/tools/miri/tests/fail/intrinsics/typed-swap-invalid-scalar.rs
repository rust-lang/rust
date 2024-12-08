#![feature(core_intrinsics)]
#![feature(rustc_attrs)]

use std::intrinsics::typed_swap;
use std::ptr::addr_of_mut;

fn invalid_scalar() {
    let mut a = 1_u8;
    let mut b = 2_u8;
    unsafe {
        let a = addr_of_mut!(a).cast::<bool>();
        let b = addr_of_mut!(b).cast::<bool>();
        typed_swap(a, b); //~ERROR: constructing invalid value
    }
}

fn main() {
    invalid_scalar();
}
