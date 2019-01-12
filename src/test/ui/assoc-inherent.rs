// Test associated types are, until #8995 is implemented, forbidden in inherent impls.

struct Foo;

impl Foo {
    type Bar = isize; //~ERROR associated types are not yet supported in inherent impls (see #8995)
}

fn main() {}
