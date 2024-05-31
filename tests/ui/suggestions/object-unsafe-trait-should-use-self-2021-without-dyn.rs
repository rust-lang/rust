//@ edition:2021
#![allow(bare_trait_objects)]
trait A: Sized {
    fn f(a: A) -> A;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    //~| ERROR associated item referring to unboxed trait object for its own trait
}
trait B {
    fn f(b: B) -> B;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    //~| ERROR associated item referring to unboxed trait object for its own trait
}
trait C {
    fn f(&self, c: C) -> C;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    //~| ERROR associated item referring to unboxed trait object for its own trait
}

fn main() {}
