struct Foo;

impl<T> Foo {}
//~^ ERROR: the type parameter `T` is not constrained

impl<T> Default for Foo {
    //~^ ERROR: the type parameter `T` is not constrained
    fn default() -> Self { Foo }
}

// This one isn't an error
fn foo<T>() {}

fn main() {}
