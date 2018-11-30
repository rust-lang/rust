// compile-pass
// aux-build:derive-helper-shadowed.rs
// aux-build:derive-helper-shadowed-2.rs

#[macro_use]
extern crate derive_helper_shadowed;
#[macro_use(my_attr)]
extern crate derive_helper_shadowed_2;

macro_rules! my_attr { () => () }

#[derive(MyTrait)]
#[my_attr] // OK
struct S;

fn main() {}
