// Regression test. This failed before applying `#[rustc_no_writable]` to `mem::forget`, `ManuallyDrop::new`, and `MaybeDangling::new`.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes

use std::mem;

// This function is taken from the crate `derive_more`, from the file `into.rs`
unsafe fn transmute<From, To>(from: From) -> To {
    let to = unsafe { mem::transmute_copy(&from) };
    mem::forget(from);
    to
}

fn main() {
    let mut val = 10u32;
    let r: &mut u32 = &mut val;

    let to: &mut i32 = unsafe { transmute(r) };

    assert_eq!(*to, 10);
}
