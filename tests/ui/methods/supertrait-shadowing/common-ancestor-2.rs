//@ run-pass
//@ check-run-results

#![feature(supertrait_item_shadowing)]
#![warn(resolving_to_items_shadowing_supertrait_items)]
#![warn(shadowing_supertrait_items)]
#![allow(dead_code)]

trait A {
    fn hello(&self) {
        println!("A");
    }
}
impl<T> A for T {}

trait B {
    fn hello(&self) {
        println!("B");
    }
}
impl<T> B for T {}

trait C: A + B {
    fn hello(&self) {
        //~^ WARN trait item `hello` from `C` shadows identically named item
        println!("C");
    }
}
impl<T> C for T {}

fn main() {
    ().hello();
    //~^ WARN trait item `hello` from `C` shadows identically named item from supertrait
}
