#![allow(dead_code)]

#[repr(align(16.0))] //~ ERROR: invalid `repr(align)` attribute: not an unsuffixed integer
struct A(i32);

#[repr(align(15))] //~ ERROR: invalid `repr(align)` attribute: not a power of two
struct B(i32);

#[repr(align(4294967296))] //~ ERROR: invalid `repr(align)` attribute: larger than 2^29
struct C(i32);

#[repr(align(536870912))] // ok: this is the largest accepted alignment
struct D(i32);

fn main() {}
