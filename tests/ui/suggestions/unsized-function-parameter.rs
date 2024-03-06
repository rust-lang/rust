//@ run-rustfix

#![allow(dead_code, unused_variables)]

fn foo1(bar: str) {}
//~^ ERROR the size for values of type `str` cannot be known at compilation time
//~| HELP the trait `Sized` is not implemented for `str`
//~| HELP unsized fn params are gated as an unstable feature
//~| HELP function arguments must have a statically known size, borrowed types always have a known size

fn foo2(_bar: str) {}
//~^ ERROR the size for values of type `str` cannot be known at compilation time
//~| HELP the trait `Sized` is not implemented for `str`
//~| HELP unsized fn params are gated as an unstable feature
//~| HELP function arguments must have a statically known size, borrowed types always have a known size

fn foo3(_: str) {}
//~^ ERROR the size for values of type `str` cannot be known at compilation time
//~| HELP the trait `Sized` is not implemented for `str`
//~| HELP unsized fn params are gated as an unstable feature
//~| HELP function arguments must have a statically known size, borrowed types always have a known size

fn main() {}
