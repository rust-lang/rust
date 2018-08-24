// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.

#![feature(custom_attribute)]
#![deny(unused_attributes)]

#![mutable_doc] //~ ERROR unused attribute

#[dance] mod a {} //~ ERROR unused attribute

#[dance] fn main() {} //~ ERROR unused attribute
