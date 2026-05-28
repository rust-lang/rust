//@ edition:2015
//@ check-pass
// Make sure that we don't eagerly recover `async ::Bound` in edition 2015.

mod async {
    pub trait Foo {}
}

fn test(x: impl async ::Foo) {}

fn main() {}
