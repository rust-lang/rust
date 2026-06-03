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

trait B: A {
    fn hello(&self) -> &'static str {
        //~^ WARN trait item `hello` from `B` shadows identically named item
        "B"
    }
}
impl<T> B for T {}

fn main() {
    assert_eq!(().hello(), "B");
    //~^ WARN trait item `hello` from `B` shadows identically named item from supertrait
}
