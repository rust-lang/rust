#![feature(intrinsics)]

#[rustc_intrinsic]
unsafe fn foo();
//~^ ERROR E0093

fn main() {}
