struct Baz { q: Option<Foo> }
//~^ ERROR recursive types `Baz` and `Foo` have infinite size

struct Foo { q: Option<Baz> }

impl Foo { fn bar(&self) {} }

fn main() {}
