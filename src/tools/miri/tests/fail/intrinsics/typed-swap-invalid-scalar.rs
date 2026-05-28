//@revisions: left right
#![feature(core_intrinsics)]
#![feature(rustc_attrs)]

use std::intrinsics::typed_swap_nonoverlapping;
use std::ptr::addr_of_mut;

fn invalid_scalar() {
    // We run the test twice, with either the left or the right side being invalid.
    let mut a = if cfg!(left) { 2_u8 } else { 1_u8 };
    let mut b = if cfg!(right) { 3_u8 } else { 1_u8 };
    unsafe {
        let a = addr_of_mut!(a).cast::<bool>();
        let b = addr_of_mut!(b).cast::<bool>();
        typed_swap_nonoverlapping(a, b); //~ERROR: constructing invalid value
    }
}

fn main() {
    invalid_scalar();
}
