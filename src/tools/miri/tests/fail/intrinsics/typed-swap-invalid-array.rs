#![feature(core_intrinsics)]
#![feature(rustc_attrs)]

use std::intrinsics::typed_swap_nonoverlapping;
use std::ptr::addr_of_mut;

fn invalid_array() {
    let mut a = [1_u8; 100];
    let mut b = [2_u8; 100];
    unsafe {
        let a = addr_of_mut!(a).cast::<[bool; 100]>();
        let b = addr_of_mut!(b).cast::<[bool; 100]>();
        typed_swap_nonoverlapping(a, b); //~ERROR: constructing invalid value
    }
}

fn main() {
    invalid_array();
}
