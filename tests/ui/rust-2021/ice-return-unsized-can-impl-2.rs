// Doesn't trigger ICE when returning unsized trait that can be impl
// issue https://github.com/rust-lang/rust/issues/125512
//@ edition:2021

trait B {
    fn f(a: A) -> A;
    //~^ ERROR: expected a type, found a trait
    //~| ERROR: expected a type, found a trait
}

trait A {
    fn concrete(b: B) -> B;
    //~^ ERROR: expected a type, found a trait
    //~| ERROR: expected a type, found a trait
}

fn main() {}
