// build-pass (FIXME(62277): could be check-pass?)

// Some constants that *are* valid
#![feature(const_transmute)]

use std::mem;
use std::ptr::NonNull;
use std::num::{NonZeroU8, NonZeroUsize};

const NON_NULL_PTR1: NonNull<u8> = unsafe { mem::transmute(1usize) };
const NON_NULL_PTR2: NonNull<u8> = unsafe { mem::transmute(&0) };

const NON_NULL_U8: NonZeroU8 = unsafe { mem::transmute(1u8) };
const NON_NULL_USIZE: NonZeroUsize = unsafe { mem::transmute(1usize) };

const UNIT: () = ();

fn main() {}
