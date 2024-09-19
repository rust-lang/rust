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

trait B {
    fn hello(&self) {
        println!("B");
    }
}
impl<T> B for T {}

trait C: A + B {
    fn hello(&self) {
        println!("C");
    }
}
impl<T> C for T {}

// `D` extends `C` which extends `B` and `A`

trait D: C {
    fn hello(&self) {
        println!("D");
    }
}
impl<T> D for T {}

fn main() {
    ().hello();
    //~^ WARN trait method `hello` from `D` shadows identically named method from supertrait
}
