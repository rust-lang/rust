//@ known-bug: #120241
//@ edition:2021
#![feature(dyn_compatible_for_dispatch)]

trait B {
    fn f(a: A) -> A;
}

trait A {
    fn g(b: B) -> B;
}

fn main() {}
