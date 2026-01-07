//@ run-pass
#![allow(unconstructable_pub_struct)]

pub struct Z(#[allow(dead_code)] &'static Z);

pub fn main() {}
