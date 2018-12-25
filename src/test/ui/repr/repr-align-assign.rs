// run-rustfix

#![allow(dead_code)]

#[repr(align=8)] //~ ERROR incorrect `repr(align)` attribute format
struct A(u64);

#[repr(align="8")] //~ ERROR incorrect `repr(align)` attribute format
struct B(u64);

fn main() {}
