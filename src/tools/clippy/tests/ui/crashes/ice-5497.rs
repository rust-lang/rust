// reduced from rustc issue-69020-assoc-const-arith-overflow.rs
#![allow(clippy::out_of_bounds_indexing)]

pub fn main() {}

pub trait Foo {
    const OOB: i32;
}

impl<T: Foo> Foo for Vec<T> {
    const OOB: i32 = [1][1] + T::OOB;
    //~^ ERROR: operation will panic
}
