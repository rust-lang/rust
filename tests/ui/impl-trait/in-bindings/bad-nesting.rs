#![feature(impl_trait_in_bindings)]

trait Foo<T> {}

impl Foo<()> for () {}

fn main() {
    let x: impl Foo<impl Sized> = ();
    //~^ ERROR nested `impl Trait` is not allowed
}
