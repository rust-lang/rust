// Simple test that checks if the writeable flag works.
//@compile-flags: -Zmiri-tree-borrows

#![feature(rustc_attrs)]
#![allow(internal_features)]

fn main() {
    let mut x = 1;
    foo(&mut x);
    assert_eq!(x, 2);
}

#[rustc_no_writable]
fn foo(x: &mut i32) {
    *x += 1;
}
