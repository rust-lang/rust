//@ known-bug: rust-lang/rust#125512
//@ edition:2021
#![feature(dyn_compatible_for_dispatch)]
trait B {
    fn f(a: A) -> A;
}
trait A {
    fn concrete(b: B) -> B;
}
fn main() {}
