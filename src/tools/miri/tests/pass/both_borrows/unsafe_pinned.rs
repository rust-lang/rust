//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(unsafe_pinned)]

use std::pin::UnsafePinned;

fn mutate(x: &UnsafePinned<i32>) {
    let ptr = x as *const _ as *mut i32;
    unsafe { ptr.write(42) };
}

fn main() {
    let x = UnsafePinned::new(0);
    mutate(&x);
    assert_eq!(x.into_inner(), 42);
}
