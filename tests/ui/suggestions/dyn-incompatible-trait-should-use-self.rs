#![allow(bare_trait_objects)]
trait A: Sized {
    fn f(a: A) -> A;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `A` is not dyn compatible
}
trait B {
    fn f(a: B) -> B;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `B` is not dyn compatible
}
trait C {
    fn f(&self, a: C) -> C;
}

fn main() {}
