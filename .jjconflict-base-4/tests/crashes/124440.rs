//@ known-bug: rust-lang/rust#124440

#![allow(warnings)]

trait Foo {}

impl<F> Foo for F where F: FnMut(&()) {}

struct Bar<F> {
    f: F,
}

impl<F> Foo for Bar<F> where F: Foo {}

fn assert_foo<F>(_: F)
where
    Bar<F>: Foo,
{
}

fn main() {
    assert_foo(|_| ());
}
