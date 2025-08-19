// ignore-tidy-linelength
#![feature(core_intrinsics)]

use std::intrinsics::{copy, copy_nonoverlapping};
use std::mem;

const COPY_ZERO: () = unsafe {
    // Since we are not copying anything, this should be allowed.
    let src = ();
    let mut dst = ();
    copy_nonoverlapping(&src as *const _ as *const u8, &mut dst as *mut _ as *mut u8, 0);
};

const COPY_OOB_1: () = unsafe {
    let mut x = 0i32;
    let dangle = (&mut x as *mut i32).wrapping_add(10);
    // Zero-sized copy is fine.
    copy_nonoverlapping(0x100 as *const i32, dangle, 0);
    // Non-zero-sized copy is not.
    copy_nonoverlapping(0x100 as *const i32, dangle, 1); //~ ERROR which is a dangling pointer
};
const COPY_OOB_2: () = unsafe {
    let x = 0i32;
    let dangle = (&x as *const i32).wrapping_add(10);
    // Zero-sized copy is fine.
    copy_nonoverlapping(dangle, 0x100 as *mut i32, 0);
    // Non-zero-sized copy is not.
    copy_nonoverlapping(dangle, 0x100 as *mut i32, 1); //~ ERROR is at or beyond the end of the allocation of size 4 bytes
};

const COPY_SIZE_OVERFLOW: () = unsafe {
    let x = 0;
    let mut y = 0;
    copy(&x, &mut y, 1usize << (mem::size_of::<usize>() * 8 - 1)); //~ ERROR overflow computing total size of `copy`
};
const COPY_NONOVERLAPPING_SIZE_OVERFLOW: () = unsafe {
    let x = 0;
    let mut y = 0;
    copy_nonoverlapping(&x, &mut y, 1usize << (mem::size_of::<usize>() * 8 - 1)); //~ ERROR overflow computing total size of `copy_nonoverlapping`
};

fn main() {}
