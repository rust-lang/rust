//@ run-pass

#![allow(unused_variables)]

// #45662

#[repr(align(16))]
pub struct A(#[allow(dead_code)] i64);

#[allow(improper_ctypes_definitions)]
pub extern "C" fn foo(x: A) {}

fn main() {
    foo(A(0));
}
