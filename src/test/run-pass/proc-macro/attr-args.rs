// aux-build:attr-args.rs

#![allow(warnings)]

extern crate attr_args;
use attr_args::{attr_with_args, identity};

#[attr_with_args(text = "Hello, world!")]
fn foo() {}

#[identity(fn main() { assert_eq!(foo(), "Hello, world!"); })]
struct Dummy;
