//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-fail
//@ known-bug: unknown

// A pointer to an unsafe binder should be thin if a pointer to the inner type
// would be thin; and wide if a pointer to the inner type would be wide.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::mem::{ManuallyDrop, size_of};
use std::unsafe_binder::unwrap_binder;

const THIN: usize = size_of::<usize>();
const WIDE: usize = 2 * size_of::<usize>();

const _: () = assert!(size_of::<&ManuallyDrop<[u8]>>() == WIDE);
const _: () = assert!(size_of::<&unsafe<> ManuallyDrop<[u8]>>() == THIN);
const _: () = assert!(size_of::<&ManuallyDrop<dyn Debug>>() == WIDE);
const _: () = assert!(size_of::<&unsafe<'a> ManuallyDrop<dyn Debug + 'a>>() == WIDE);

fn slice_to_wide(p: *const ManuallyDrop<[u8]>) -> *const unsafe<> ManuallyDrop<[u8]> {
    p as _
}

fn dyn_to_wide(
    p: *const ManuallyDrop<dyn Debug>,
) -> *const unsafe<'a> ManuallyDrop<dyn Debug + 'a> {
    p as _
}

fn len(x: &unsafe<> ManuallyDrop<[u8]>) -> usize {
    std::mem::size_of_val(x)
}

fn unwrap(q: *const unsafe<'a> ManuallyDrop<dyn Debug + 'a>) -> &'static ManuallyDrop<dyn Debug> {
    unsafe { &unwrap_binder!(*q) }
}

fn main() {
    let s: &[u8] = &[1, 2, 3];
    let _ = slice_to_wide(s as *const [u8] as *const ManuallyDrop<[u8]>);

    let x = ManuallyDrop::new(1u8);
    let _ = unwrap(dyn_to_wide(&x as &ManuallyDrop<dyn Debug>));

    let _f: fn(&unsafe<> ManuallyDrop<[u8]>) -> usize = len;
}
