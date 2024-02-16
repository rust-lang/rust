//@ edition:2021
#![allow(bare_trait_objects)]
trait A: Sized {
    fn f(a: A) -> A;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    //~| ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `A` cannot be made into an object
}
trait B {
    fn f(b: B) -> B;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    //~| ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `B` cannot be made into an object
}
trait C {
    fn f(&self, c: C) -> C;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    //~| ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `C` cannot be made into an object
}

fn main() {}
