// Test that async fn works when nested inside of
// impls with lifetime parameters.
//
//@ check-pass
//@ edition:2018

struct Foo<'a>(&'a ());

impl<'a> Foo<'a> {
    fn test() {
        async fn test() {}
    }
}

fn main() { }
