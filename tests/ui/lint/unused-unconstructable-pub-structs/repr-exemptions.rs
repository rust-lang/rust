//@ check-pass

#![deny(unused_unconstructable_pub_structs)]

#[repr(C)]
pub struct CRepr(i32);

#[repr(transparent)]
pub struct Transparent(i32);

fn main() {}
