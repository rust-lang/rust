#![feature(type_alias_impl_trait)]
//@ check-pass

use std::iter::{once, Chain};

trait Trait<'a, 'b: 'a> {}

impl<'a, 'b: 'a, T> Trait<'a, 'b> for std::iter::Cloned<T> {}

type I<'a, 'b: 'a, A: Trait<'a, 'b>> = Chain<A, impl Iterator<Item = &'static str>>;
#[define_opaque(I)]
fn test2<'a, 'b, A: Trait<'a, 'b> + Iterator<Item = &'static str>>(x: A) -> I<'a, 'b, A> {
    x.chain(once("5"))
}

fn main() {
    assert_eq!(vec!["1", "3", "5"], test2(["1", "3"].iter().cloned()).collect::<Vec<_>>());
}
