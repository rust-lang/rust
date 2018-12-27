#![allow(stable_features)]
#![allow(unused_variables)]

// #45662

#![feature(repr_align)]

#[repr(align(16))]
pub struct A(i64);

pub extern "C" fn foo(x: A) {}

fn main() {
    foo(A(0));
}
