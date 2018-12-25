// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.

#![deny(unused_attributes)]
#![allow(dead_code)]
#![feature(custom_attribute)]

#[ab_isize="stdcall"] extern {} //~ ERROR unused attribute

#[fixed_stack_segment] fn f() {} //~ ERROR unused attribute

fn main() {}
