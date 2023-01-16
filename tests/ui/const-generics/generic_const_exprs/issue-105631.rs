#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct A<const B: str = 1, C>;
//~^ ERROR generic parameters with a default must be trailing
//~| ERROR the size for values of type `str` cannot be known at compilation time
//~| ERROR mismatched types
//~| ERROR parameter `C` is never used
//~| ERROR `str` is forbidden as the type of a const generic parameter

fn main() {}
