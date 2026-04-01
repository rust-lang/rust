//! Check for compilation errors when recursive types are defined in a way
//! that leads to an infinite size.

struct Baz {
    //~^ ERROR recursive types `Baz` and `Foo` have infinite size
    q: Option<Foo>,
}
struct Foo {
    q: Option<Baz>,
}

impl Foo {
    fn bar(&self) {}
}

fn main() {}
