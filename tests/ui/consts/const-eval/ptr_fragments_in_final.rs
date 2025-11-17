//! Test that we properly error when there is a pointer fragment in the final value.

use std::{mem::{self, MaybeUninit}, ptr};

type Byte = MaybeUninit<u8>;

const unsafe fn memcpy(dst: *mut Byte, src: *const Byte, n: usize) {
    let mut i = 0;
    while i < n {
        dst.add(i).write(src.add(i).read());
        i += 1;
    }
}

const MEMCPY_RET: MaybeUninit<*const i32> = unsafe { //~ERROR: partial pointer in final value
    let ptr = &42;
    let mut ptr2 = MaybeUninit::new(ptr::null::<i32>());
    memcpy(&mut ptr2 as *mut _ as *mut _, &ptr as *const _ as *const _, mem::size_of::<&i32>() / 2);
    // Return in a MaybeUninit so it does not get treated as a scalar.
    ptr2
};

// Mixing two different pointers that have the same provenance.
const MIXED_PTR: MaybeUninit<*const u8> = { //~ERROR: partial pointer in final value
    static A: u8 = 123;
    const HALF_PTR: usize = std::mem::size_of::<*const ()>() / 2;

    unsafe {
        let x: *const u8 = &raw const A;
        let mut y = MaybeUninit::new(x.wrapping_add(usize::MAX / 4));
        core::ptr::copy_nonoverlapping(
            (&raw const x).cast::<u8>(),
            (&raw mut y).cast::<u8>(),
            HALF_PTR,
        );
        y
    }
};

fn main() {}
