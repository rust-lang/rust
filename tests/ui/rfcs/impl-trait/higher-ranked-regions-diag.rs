// Regression test for #97099.
// This was an ICE because `impl Sized` captures the lifetime 'a.

trait Trait<E> {
    type Assoc;
}

struct Foo;

impl<'a> Trait<&'a ()> for Foo {
    type Assoc = ();
}

fn foo() -> impl for<'a> Trait<&'a ()> {
    Foo
}

fn bar() -> impl for<'a> Trait<&'a (), Assoc = impl Sized> {
    foo()
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn main() {}
