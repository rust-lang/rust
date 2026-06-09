//@ run-pass

#![feature(supertrait_item_shadowing)]
#![warn(resolving_to_items_shadowing_supertrait_items)]
#![warn(shadowing_supertrait_items)]
#![allow(dead_code)]

trait A {
    fn hello(&self) -> &'static str {
        "A"
    }
}
impl<T> A for T {}

trait B {
    fn hello(&self) -> &'static str {
        "B"
    }
}
impl<T> B for T {}

trait C: A + B {
    fn hello(&self) -> &'static str {
        //~^ WARN trait item `hello` from `C` shadows identically named item
        "C"
    }
}
impl<T> C for T {}

// `D` extends `C` which extends `B` and `A`

trait D: C {
    fn hello(&self) -> &'static str {
        //~^ WARN trait item `hello` from `D` shadows identically named item
        "D"
    }
}
impl<T> D for T {}

fn main() {
    assert_eq!(().hello(), "D");
    //~^ WARN trait item `hello` from `D` shadows identically named item from supertrait
}
