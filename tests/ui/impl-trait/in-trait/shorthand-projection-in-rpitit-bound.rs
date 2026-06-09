//@ check-pass

// Ensure that we can resolve a shorthand projection in an item bound in an RPITIT.

pub trait Bar {
    type Foo;
}

pub trait Baz {
    fn boom<X: Bar>() -> impl Bar<Foo = X::Foo>;
}

fn main() {}
