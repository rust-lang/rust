// check-pass

// revisions: min full
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full, feature(type_alias_impl_trait))]
//[full]~^ WARN incomplete

use std::iter::{once, Chain};

fn test1<A: Iterator<Item = &'static str>>(x: A) -> Chain<A, impl Iterator<Item = &'static str>> {
    x.chain(once(","))
}

type I<A> = Chain<A, impl Iterator<Item = &'static str>>;
fn test2<A: Iterator<Item = &'static str>>(x: A) -> I<A> {
    x.chain(once(","))
}

fn main() {}
