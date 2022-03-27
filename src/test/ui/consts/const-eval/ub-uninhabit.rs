// stderr-per-bitwidth
#![allow(const_err)] // make sure we cannot allow away the errors tested here

use std::mem;

#[derive(Copy, Clone)]
enum Bar {}

#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}

const BAD_BAD_BAD: Bar = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR evaluation of constant value failed

const BAD_BAD_REF: &Bar = unsafe { mem::transmute(1usize) };
//~^ ERROR evaluation of constant value failed

const BAD_BAD_ARRAY: [Bar; 1] = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR evaluation of constant value failed

fn main() {}
