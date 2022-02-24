// compile-flags: -Z chalk-migration

#![feature(trivial_bounds)]

trait Bar {
    fn foo();
}
trait Foo: Bar { }

struct S where S: Foo;

impl Foo for S { //~ ERROR the trait bound `S: Bar` is not satisfied
}

fn bar<T: Bar>() {
    T::foo();
}

fn foo<T: Foo>() {
    bar::<T>()
}

fn main() {
    // For some reason, the error is duplicated...

    foo::<S>()
}
