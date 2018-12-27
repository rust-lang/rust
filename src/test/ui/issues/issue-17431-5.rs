use std::marker;

struct Foo { foo: Bar<Foo> }

struct Bar<T> { x: Bar<Foo> , marker: marker::PhantomData<T> }
//~^ ERROR recursive type `Bar` has infinite size

impl Foo { fn foo(&self) {} }

fn main() {
}
