#![feature(fn_delegation)]

fn a() {}

reuse a as b { #![rustc_dummy] self } //~ ERROR an inner attribute is not permitted in this context
//~^ ERROR: delegation's target expression is specified for function with no params
//~| ERROR: this function takes 0 arguments but 1 argument was supplied

fn main() {}
