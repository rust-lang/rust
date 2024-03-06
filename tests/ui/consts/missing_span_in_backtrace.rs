//@ compile-flags: -Z ui-testing=no


#![feature(const_swap)]
#![feature(const_mut_refs)]
use std::{
    mem::{self, MaybeUninit},
    ptr,
};

const X: () = {
    let mut ptr1 = &1;
    let mut ptr2 = &2;

    // Swap them, bytewise.
    unsafe {
        ptr::swap_nonoverlapping(
            &mut ptr1 as *mut _ as *mut MaybeUninit<u8>,
            &mut ptr2 as *mut _ as *mut MaybeUninit<u8>,
            mem::size_of::<&i32>(),
        );
    }
};

fn main() {
    X
}
