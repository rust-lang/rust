//@ check-pass
//@ edition: 2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![deny(refining_impl_trait)]

pub trait Foo {
    type Item;

    fn hello() -> impl Iterator<Item = Self::Item>;
}

impl Foo for () {
    type Item = ();

    fn hello() -> impl Iterator<Item = ()> { [()].into_iter() }
}

fn main() {}
