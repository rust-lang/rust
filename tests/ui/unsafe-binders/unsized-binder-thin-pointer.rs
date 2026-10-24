//@ compile-flags: -Znext-solver
//@ check-fail

// A pointer to an unsafe binder should be thin if a pointer to the inner type
// would be thin; and wide if a pointer to the inner type would be wide.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::mem::{ManuallyDrop, size_of};
use std::ptr::Thin;

const THIN: usize = size_of::<usize>();
const WIDE: usize = 2 * size_of::<usize>();

const _: () = assert!(size_of::<&unsafe<'a> &'a [u8]>() == THIN);
const _: () = assert!(size_of::<&unsafe<'a> (&'a u8, u32)>() == THIN);
const _: () = assert!(size_of::<&ManuallyDrop<[u8]>>() == WIDE);
const _: () = assert!(size_of::<&unsafe<> ManuallyDrop<[u8]>>() == WIDE);
const _: () = assert!(size_of::<&ManuallyDrop<dyn Debug>>() == WIDE);
const _: () = assert!(size_of::<&unsafe<'a> ManuallyDrop<dyn Debug + 'a>>() == WIDE);

fn is_thin<T: ?Sized + Thin>() {}

fn thin() {
    is_thin::<unsafe<'a> &'a [u8]>();
    is_thin::<unsafe<'a> (&'a u8, u32)>();
}

fn wide() {
    is_thin::<unsafe<> ManuallyDrop<[u8]>>(); //~ ERROR the trait bound
    is_thin::<unsafe<'a> ManuallyDrop<dyn Debug + 'a>>(); //~ ERROR the trait bound
}

fn main() {}
