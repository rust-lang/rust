#![feature(intrinsics)]
#![feature(rustc_attrs)]

#[rustc_intrinsic]
fn size_of<T>();
//~^ ERROR E0308

fn main() {}
