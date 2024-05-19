use std::marker;

struct Foo<T> { foo: Option<Option<Foo<T>>>, marker: marker::PhantomData<T> }
//~^ ERROR recursive type `Foo` has infinite size

impl<T> Foo<T> { fn bar(&self) {} }

fn main() {}
