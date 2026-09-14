#![feature(fn_delegation)]

mod to_reuse {
    pub fn foo(x: usize) -> usize { x }
    pub fn bar() -> () { () }
}

reuse to_reuse::{foo, bar} { self + 1 }
//~^ ERROR: this function takes 0 arguments but 1 argument was supplied
//~| ERROR: delegation's target expression is specified for function with no params

fn main() {}
