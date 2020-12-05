use std::sync::Mutex;

enum Foo { X(Mutex<Option<Foo>>) }
//~^ ERROR recursive type `Foo` has infinite size

impl Foo { fn bar(self) {} }

fn main() {}
