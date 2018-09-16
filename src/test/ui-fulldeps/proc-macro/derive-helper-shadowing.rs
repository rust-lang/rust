// aux-build:derive-helper-shadowing.rs

extern crate derive_helper_shadowing;
use derive_helper_shadowing::*;

#[my_attr] //~ ERROR `my_attr` is ambiguous
#[derive(MyTrait)]
struct S;

fn main() {}
