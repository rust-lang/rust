// build-pass (FIXME(62277): could be check-pass?)
#![deny(unused_attributes)]

#[repr(C)]
#[repr(packed)]
pub struct Foo;

#[repr(packed)]
#[repr(C)]
pub struct Bar;

fn main() { }
