//! Regression test for https://github.com/rust-lang/rust/issues/163800,
//! where the generated code always referred to `Self`, which was invalid
//! when an attribute macro rewrites the type to an enum.

//@ check-pass
//@ proc-macro: struct-to-enum.rs

extern crate struct_to_enum;
use struct_to_enum::*;

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[struct_to_enum::phantom]
struct Spooky<T>;

fn main() {}
