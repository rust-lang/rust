// Detect and reject escaping late-bound generic params in
// the type of assoc consts used in an equality bound.
#![feature(associated_const_equality)]

trait Trait<'a> {
    const K: &'a ();
}

fn take(_: impl for<'r> Trait<'r, K = { &() }>) {}
//~^ ERROR the type of the associated constant `K` cannot capture late-bound generic parameters
//~| NOTE its type cannot capture the late-bound lifetime parameter `'r`
//~| NOTE the late-bound lifetime parameter `'r` is defined here
//~| NOTE `K` has type `&'r ()`

fn main() {}
