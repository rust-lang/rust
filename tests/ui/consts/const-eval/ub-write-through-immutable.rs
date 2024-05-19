//! Ensure we catch UB due to writing through a shared reference.
#![feature(const_mut_refs, const_refs_to_cell)]
#![deny(writes_through_immutable_pointer)]
#![allow(invalid_reference_casting)]

use std::mem;
use std::cell::UnsafeCell;

const WRITE_AFTER_CAST: () = unsafe {
    let mut x = 0;
    let ptr = &x as *const i32 as *mut i32;
    *ptr = 0; //~ERROR: writes_through_immutable_pointer
    //~^ previously accepted
};

const WRITE_AFTER_TRANSMUTE: () = unsafe {
    let mut x = 0;
    let ptr: *mut i32 = mem::transmute(&x);
    *ptr = 0; //~ERROR: writes_through_immutable_pointer
    //~^ previously accepted
};

// it's okay when there is interior mutability;
const WRITE_INTERIOR_MUT: () = unsafe {
    let x = UnsafeCell::new(0);
    let ptr = &x as *const _ as *mut i32;
    *ptr = 0;
};

fn main() {}
