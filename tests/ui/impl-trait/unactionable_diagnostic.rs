//@ run-rustfix

pub trait Trait {}

pub struct Foo;

impl Trait for Foo {}

fn foo<'x, P>(
    _post: P,
    x: &'x Foo,
) -> &'x impl Trait {
    x
}

pub fn bar<'t, T>(
    //~^ HELP: consider adding an explicit lifetime bound
    post: T,
    x: &'t Foo,
) -> &'t impl Trait {
    foo(post, x)
    //~^ ERROR: the parameter type `T` may not live long enough
}

fn main() {}
