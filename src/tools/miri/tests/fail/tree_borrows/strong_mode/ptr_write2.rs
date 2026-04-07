// Tests that UB exists for explicit write, even without writable
//@compile-flags: -Zmiri-tree-borrows

#![feature(rustc_attrs)]
#![allow(internal_features)]

#[rustc_no_writable]
fn foo(x: &mut u8, y: *mut u8) -> u8 {
    let val = unsafe { *y };
    *x = 42; //~ ERROR: write access
    val
}

fn main() {
    let mut x = 0u8;
    let ptr = &raw mut x;
    let res = foo(&mut x, ptr);
    assert_eq!(res, 0);
}
