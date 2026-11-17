//@revisions: speed none
#![feature(core_intrinsics)]
use std::{intrinsics, mem};

#[cfg_attr(speed, optimize(speed))]
#[cfg_attr(none, optimize(none))]
fn main() {
    unsafe {
        let ptr: *const [u8] = mem::transmute((1usize, usize::MAX));
        intrinsics::align_of_val(ptr);
        //~^ERROR: invalid metadata in wide pointer
    }
}
