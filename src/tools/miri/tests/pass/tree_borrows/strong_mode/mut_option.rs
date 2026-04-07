// This test shows the old behavior of the test in `fail/tree_borrows/strong_mode/mut_option.rs`. The only difference is the addition of the `#[rustc_no_writable]` attribute.
//@compile-flags: -Zmiri-tree-borrows

#![feature(rustc_attrs)]
#![allow(internal_features)]

fn main() {
    let mut x = 42;

    let ptr = std::ptr::from_mut(&mut x);
    let a = unsafe { &mut *ptr };
    let b = unsafe { &mut *ptr };

    let _c = foo(Some(a));
    println!("{:?}", *b);
}

#[rustc_no_writable]
fn foo(_x: Option<&mut i32>) {}
