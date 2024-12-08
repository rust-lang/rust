//@ check-pass

// Make sure that we only consider *Self* supertrait predicates
// in the `unused_must_use` lint.

#![feature(trait_alias)]
#![deny(unused_must_use)]

trait Foo<T> = Sized where T: Iterator;

fn test<T: Iterator>() -> impl Foo<T> {}

fn main() {
    test::<std::iter::Once<()>>();
}
