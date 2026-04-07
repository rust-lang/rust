// This code should work under the weak mode of tree borrows, enforced by the `rustc_no_writable` flag.
// Under the strong mode, this code no longer passes. This is tested in `fail/tree_borrows/strong_mode/as_mut_ptr.rs`. The only difference is the removal of the `rustc_no_writable` flag.
//@compile-flags: -Zmiri-tree-borrows

#![feature(rustc_attrs)]
#![allow(internal_features)]

fn main() {
    let mut x = ["one", "two", "three"];

    let ptr = std::ptr::from_mut(&mut x);
    let a = unsafe { &mut *ptr };
    let b = unsafe { &mut *ptr };

    let _c = as_mut_ptr(a);
    println!("{:?}", *b);
}

#[rustc_no_writable]
pub const fn as_mut_ptr(x: &mut [&str; 3]) -> *mut str {
    x as *mut [&str] as *mut str
}
