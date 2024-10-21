//@ run-pass
//@ check-run-results

#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

trait A {
    fn hello(&self) {
        println!("A");
    }
}
impl<T> A for T {}

trait B: A {
    fn hello(&self) {
        println!("B");
    }
}
impl<T> B for T {}

fn main() {
    ().hello();
    //~^ WARN trait method `hello` from `B` shadows identically named method from supertrait
}
