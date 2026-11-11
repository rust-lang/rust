//@ run-pass

#![feature(type_alias_impl_trait)]

use std::iter::{Chain, once};

type I<A> = Chain<A, impl Iterator<Item = &'static str>>;
#[define_opaque(I)]
fn test2<A: Iterator<Item = &'static str>>(x: A) -> I<A> {
    x.chain(once("5"))
}

fn main() {
    assert_eq!(vec!["1", "3", "5"], test2(["1", "3"].iter().cloned()).collect::<Vec<_>>());
}
