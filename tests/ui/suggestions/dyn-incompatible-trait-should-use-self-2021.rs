//@ edition:2021
#![allow(bare_trait_objects)]
trait A: Sized {
    fn f(a: dyn A) -> dyn A;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `A` cannot be made into an object
}
trait B {
    fn f(a: dyn B) -> dyn B;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `B` cannot be made into an object
}
trait C {
    fn f(&self, a: dyn C) -> dyn C;
}

fn main() {}
