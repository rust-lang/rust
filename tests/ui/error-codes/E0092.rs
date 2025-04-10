#![feature(intrinsics)]

#[rustc_intrinsic]
unsafe fn atomic_foo(); //~ ERROR E0092

fn main() {}
