#![allow(bare_trait_objects)]
trait A: Sized {
    fn f(a: A) -> A;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `A` cannot be made into an object
}
trait B {
    fn f(a: B) -> B;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `B` cannot be made into an object
}
trait C {
    fn f(&self, a: C) -> C;
}

fn main() {}
