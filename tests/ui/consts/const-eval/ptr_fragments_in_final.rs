//! Test that we properly error when there is a pointer fragment in the final value.
//@ ignore-test: disabled due to <https://github.com/rust-lang/rust/issues/146291>

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

fn main() {
    assert_eq!(unsafe { MEMCPY_RET.assume_init().read() }, 42);
}
