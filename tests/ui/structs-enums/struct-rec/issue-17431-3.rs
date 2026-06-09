use std::sync::Mutex;

struct Foo { foo: Mutex<Option<Foo>> }
//~^ ERROR recursive type `Foo` has infinite size

impl Foo { fn bar(&self) {} }

fn main() {}
