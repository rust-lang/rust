//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
//@ known-bug: unknown

// An unsafe binder should only be sized if the inner type is sized.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::mem::ManuallyDrop;

fn requires_sized<T: Sized>() {}

fn sized_bound() {
    requires_sized::<unsafe<> ManuallyDrop<[u8]>>();
    requires_sized::<unsafe<'a> ManuallyDrop<dyn Debug + 'a>>();
}

fn by_value(x: unsafe<> ManuallyDrop<[u8]>) -> unsafe<> ManuallyDrop<[u8]> {
    x
}

fn move_out_of_box(x: Box<unsafe<> ManuallyDrop<[u8]>>) {
    let _y = *x;
}

fn size<T>() -> usize {
    std::mem::size_of::<T>()
}

fn main() {
    sized_bound();
    let _f: fn(unsafe<> ManuallyDrop<[u8]>) -> unsafe<> ManuallyDrop<[u8]> = by_value;
    let _g: fn(Box<unsafe<> ManuallyDrop<[u8]>>) = move_out_of_box;
    size::<unsafe<> ManuallyDrop<[u8]>>();
}
