#![feature(impl_trait_in_bindings)]

trait Foo {}

struct W<T>(T);
impl<T> Foo for W<T> where T: Foo {}

fn main() {
    let x: impl Foo = W(());
    //~^ ERROR the trait bound `(): Foo` is not satisfied
    let x: W<impl Foo> = W(());
    //~^ ERROR the trait bound `(): Foo` is not satisfied
}
