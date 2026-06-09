//@ check-pass

#![feature(type_alias_impl_trait)]

use std::iter::{once, Chain};

fn test1<A: Iterator<Item = &'static str>>(x: A) -> Chain<A, impl Iterator<Item = &'static str>> {
    x.chain(once(","))
}

type I<A> = Chain<A, impl Iterator<Item = &'static str>>;
#[define_opaque(I)]
fn test2<A: Iterator<Item = &'static str>>(x: A) -> I<A> {
    x.chain(once(","))
}

fn main() {}
