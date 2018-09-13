// aux-build:derive-helper-shadowing.rs

extern crate derive_helper_shadowing;
use derive_helper_shadowing::*;

#[derive(MyTrait)]
#[my_attr] //~ ERROR `my_attr` is ambiguous
struct S;

fn main() {}
