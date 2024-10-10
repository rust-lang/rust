//@ edition:2021
#![allow(bare_trait_objects)]
trait A: Sized {
    fn f(a: A) -> A;
    //~^ ERROR expected a type, found a trait
    //~| ERROR expected a type, found a trait
    //~| ERROR associated item referring to unboxed trait object for its own trait
}
trait B {
    fn f(b: B) -> B;
    //~^ ERROR expected a type, found a trait
    //~| ERROR expected a type, found a trait
    //~| ERROR associated item referring to unboxed trait object for its own trait
}
trait C {
    fn f(&self, c: C) -> C;
    //~^ ERROR expected a type, found a trait
    //~| ERROR expected a type, found a trait
    //~| ERROR associated item referring to unboxed trait object for its own trait
}

fn main() {}
