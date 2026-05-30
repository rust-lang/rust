#![allow(dead_code)]

#[repr(align=8)] //~ ERROR malformed `repr` attribute input
struct A(u64);

#[repr(align="8")] //~ ERROR malformed `repr` attribute input
struct B(u64);

fn main() {}
