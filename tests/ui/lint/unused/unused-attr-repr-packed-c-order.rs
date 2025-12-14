//! regression test for <https://github.com/rust-lang/rust/issues/26646>
//@ check-pass
#![deny(unused_attributes)]

#[repr(C)]
#[repr(packed)]
pub struct Foo;

#[repr(packed)]
#[repr(C)]
pub struct Bar;

fn main() {}
