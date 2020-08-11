#![feature(min_const_generics)]
#![crate_type="lib"]

struct A<const N: usize = 3>;
//~^ ERROR default values for
