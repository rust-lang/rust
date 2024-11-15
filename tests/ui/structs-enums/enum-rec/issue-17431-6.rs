use std::cell::UnsafeCell;

enum Foo { X(UnsafeCell<Option<Foo>>) }
//~^ ERROR recursive type `Foo` has infinite size
//~| ERROR cycle detected

impl Foo { fn bar(self) {} }

fn main() {}
