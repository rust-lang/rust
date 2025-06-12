trait A: Sized {
    fn f(a: dyn A) -> dyn A;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `A` is not dyn compatible
}
trait B {
    fn f(a: dyn B) -> dyn B;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `B` is not dyn compatible
}
trait C {
    fn f(&self, a: dyn C) -> dyn C;
}

fn main() {}
