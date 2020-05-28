// FIXME(chalk): should fail, see comments
// check-pass
// compile-flags: -Z chalk

#![feature(trivial_bounds)]

trait Bar {
    fn foo();
}
trait Foo: Bar { }

struct S where S: Foo;
//~^ WARN Trait bound S: Foo does not depend on any type or lifetime parameters

impl Foo for S {
}

fn bar<T: Bar>() {
    T::foo();
}

fn foo<T: Foo>() {
    bar::<T>()
}

fn main() {
    // For some reason, the error is duplicated...

    // FIXME(chalk): this order of this duplicate error seems non-determistic
    // and causes test to fail
    /*
    foo::<S>() // ERROR the type `S` is not well-formed (chalk)
    //^ ERROR the type `S` is not well-formed (chalk)
    */
}
