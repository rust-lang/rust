// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.

#![feature(custom_inner_attributes)]

#![mutable_doc] //~ ERROR attribute `mutable_doc` is currently unknown

#[dance] mod a {} //~ ERROR attribute `dance` is currently unknown

#[dance] fn main() {} //~ ERROR attribute `dance` is currently unknown
