// build-pass (FIXME(62277): could be check-pass?)
// compile-flags: -Z parse-only

#![feature(generic_associated_types)]

impl<T> Baz for T where T: Foo {
    type Quux<'a> = <T as Foo>::Bar<'a, 'static>;
}

fn main() {}
