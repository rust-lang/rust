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

// `D` extends `C` which extends `B` and `A`

trait D: C {
    fn hello(&self) {
        //~^ WARN trait item `hello` from `D` shadows identically named item
        println!("D");
    }
}
impl<T> D for T {}

fn main() {
    ().hello();
    //~^ WARN trait item `hello` from `D` shadows identically named item from supertrait
}
