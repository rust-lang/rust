#![feature(supertrait_item_shadowing)]

struct W<T>(T);

trait Upstream {
    fn hello(&self) {}
}
impl<T> Upstream for T {}

trait Downstream: Upstream {
    fn hello(&self) {}
}
impl<T> Downstream for W<T> where T: Foo {}

trait Foo {}

fn main() {
    let x = W(Default::default());
    x.hello();
    //~^ ERROR the trait bound `i32: Foo` is not satisfied
    //~| WARN trait method `hello` from `Downstream` shadows identically named method from supertrait
    let _: i32 = x.0;
}
