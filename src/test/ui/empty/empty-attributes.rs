#![deny(unused_attributes)]
#![allow()] //~ ERROR unused attribute
#![warn()] //~ ERROR unused attribute
#![deny()] //~ ERROR unused attribute
#![forbid()] //~ ERROR unused attribute
#![feature()] //~ ERROR unused attribute

#[repr()] //~ ERROR unused attribute
pub struct S;

#[target_feature()] //~ ERROR unused attribute
pub unsafe fn foo() {}

fn main() {}
