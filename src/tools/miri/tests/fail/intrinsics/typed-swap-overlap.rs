#![feature(core_intrinsics)]
#![feature(rustc_attrs)]

use std::intrinsics::typed_swap_nonoverlapping;
use std::ptr::addr_of_mut;

fn main() {
    let mut a = 0_u8;
    unsafe {
        let a = addr_of_mut!(a);
        typed_swap_nonoverlapping(a, a); //~ERROR: called on overlapping ranges
    }
}
