trait Trait {}

struct Foo;

impl Trait for Foo {}

fn foo<'t, P>(
    post: P,
    x: &'t Foo,
) -> &'t impl Trait {
    x
}

fn bar<'t, T>(
    post: T,
    x: &'t Foo,
) -> &'t impl Trait {
    foo(post, x)
    //~^ ERROR: the opaque type `foo<T>::{opaque#0}` may not live long enough
    //~| HELP: consider adding an explicit lifetime bound `foo<T>::{opaque#0}: 't`
}

fn main() {}
