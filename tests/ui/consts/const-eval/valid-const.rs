//@ check-pass
//
// Some constants that *are* valid

use std::mem;
use std::ptr::NonNull;
use std::num::NonZero;

const NON_NULL_PTR1: NonNull<u8> = unsafe { mem::transmute(1usize) };
const NON_NULL_PTR2: NonNull<u8> = unsafe { mem::transmute(&0) };

const NON_NULL_U8: NonZero<u8> = unsafe { mem::transmute(1u8) };
const NON_NULL_USIZE: NonZero<usize> = unsafe { mem::transmute(1usize) };

const UNIT: () = ();

fn main() {}
