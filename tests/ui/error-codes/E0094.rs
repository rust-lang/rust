#![feature(intrinsics)]

#[rustc_intrinsic]
fn size_of<T, U>() -> usize;
//~^ ERROR E0094

fn main() {}
