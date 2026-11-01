#![feature(impl_trait_in_bindings)]

trait Foo<'a> {
    type Out;
}

impl<'a> Foo<'a> for () {
    type Out = ();
}

fn main() {
    let x: &dyn for<'a> Foo<'a, Out = impl Sized + 'a> = &();
    //~^ ERROR cannot capture late-bound lifetime in `impl Trait` in binding
}
