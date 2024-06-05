struct Foo;

impl<const N: usize> Foo {}
//~^ ERROR: the const parameter `N` is not constrained

impl<const N: usize> Default for Foo {
    //~^ ERROR: the const parameter `N` is not constrained
    fn default() -> Self { Foo }
}

// This one isn't an error
fn foo<const N: usize>() {}

fn main() {}
