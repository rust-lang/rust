//@ check-pass
//@ compile-flags: -Z parse-crate-root-only

impl<T> Baz for T where T: Foo {
    type Quux<'a> = <T as Foo>::Bar<'a, 'static>;
}

fn main() {}
