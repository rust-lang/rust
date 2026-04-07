// Shows that without spurious writes, `tests/fail/tree_borrows/strong_mode/ptr_write1.rs` would pass
//@compile-flags: -Zmiri-tree-borrows

#![feature(rustc_attrs)]
#![allow(internal_features)]

#[rustc_no_writable]
fn foo(_x: &mut u8, y: *mut u8) -> u8 {
    // spurious write inserted here for x
    let val = unsafe { *y };
    // *x = 42; // we'd like to add this write, so there must already be UB without it because there sure is with it.
    val
}

fn main() {
    let mut x = 0u8;
    let ptr = &raw mut x;
    let res = foo(&mut x, ptr);
    assert_eq!(res, 0);
}
