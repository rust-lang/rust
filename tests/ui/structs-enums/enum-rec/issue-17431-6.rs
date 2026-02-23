use std::cell::UnsafeCell;

enum Foo {
    //~^ ERROR recursive type `Foo` has infinite size
    X(UnsafeCell<Option<Foo>>),
}

impl Foo {
    fn bar(self) {}
}

fn main() {}
