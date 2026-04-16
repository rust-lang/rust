//@ compile-flags: -Znext-solver=globally
// ICE regression test for https://github.com/rust-lang/rust/issues/151327

#![feature(min_specialization)]

trait Foo { //~ ERROR cycle detected when coherence checking all impls of trait `Foo`
    type Item;
}

trait Baz {}

impl<'a, T> Foo for &'a T
where
    Self::Item: 'a,
{
}

impl<'a, T> Foo for &T
where
    Self::Item: Baz,
{
}

fn main() {}
