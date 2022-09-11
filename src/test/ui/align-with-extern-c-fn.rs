// run-pass

#![allow(stable_features)]
#![allow(unused_variables)]

// #45662

#![feature(repr_align)]

#[repr(align(16))]
pub struct A(#[allow(unused_tuple_struct_fields)] i64);

#[allow(improper_ctypes_definitions)]
pub extern "C" fn foo(x: A) {}

fn main() {
    foo(A(0));
}
