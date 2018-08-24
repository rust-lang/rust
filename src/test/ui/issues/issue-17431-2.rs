struct Baz { q: Option<Foo> }
//~^ ERROR recursive type `Baz` has infinite size

struct Foo { q: Option<Baz> }
//~^ ERROR recursive type `Foo` has infinite size

impl Foo { fn bar(&self) {} }

fn main() {}
