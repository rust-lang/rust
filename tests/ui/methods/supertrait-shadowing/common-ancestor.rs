//@ run-pass
//@ check-run-results

#![feature(supertrait_item_shadowing)]
#![warn(supertrait_item_shadowing_usage)]
#![warn(supertrait_item_shadowing_definition)]
#![allow(dead_code)]

trait A {
    fn hello(&self) {
        println!("A");
    }
}
impl<T> A for T {}

trait B: A {
    fn hello(&self) {
        //~^ WARN trait item `hello` from `B` shadows identically named item
        println!("B");
    }
}
impl<T> B for T {}

fn main() {
    ().hello();
    //~^ WARN trait item `hello` from `B` shadows identically named item from supertrait
}
