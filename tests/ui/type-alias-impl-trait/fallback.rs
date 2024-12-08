// Tests that we correctly handle opaque types being used opaquely,
// even within their defining scope.
//
#![feature(type_alias_impl_trait)]

type Foo = impl Copy;

enum Wrapper<T> {
    First(T),
    Second
}

// This method constrains `Foo` to be `bool`
fn constrained_foo() -> Foo {
    true
}


// This method does not constrain `Foo`.
// Per RFC 2071, function bodies may either
// fully constrain an opaque type, or place no
// constraints on it.
fn unconstrained_foo() -> Wrapper<Foo> {
    Wrapper::Second
    //~^ ERROR: type annotations needed
}

fn main() {}
