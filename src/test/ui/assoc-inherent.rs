// Test associated types are forbidden in inherent impls.

struct Foo;

impl Foo {
    type Bar = isize; //~ERROR associated types are not allowed in inherent impls
}

fn main() {}
