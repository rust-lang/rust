struct Foo { foo: Option<Option<Foo>> }
//~^ ERROR recursive type `Foo` has infinite size

impl Foo { fn bar(&self) {} }

fn main() {}
