//@ check-pass

use std::iter::FromIterator;

struct DynamicAlt<P>(P);

impl<P> FromIterator<P> for DynamicAlt<P> {
    fn from_iter<T: IntoIterator<Item = P>>(_iter: T) -> Self {
        loop {}
    }
}

fn owned_context<I, F>(_: F) -> impl FnMut(I) -> I {
    |i| i
}

trait Parser<I> {}

impl<T, I> Parser<I> for T where T: FnMut(I) -> I {}

fn alt<I, P: Parser<I>>(_: DynamicAlt<P>) -> impl FnMut(I) -> I {
    |i| i
}

fn rule_to_parser<'c>() -> impl Parser<&'c str> {
    move |input| {
        let v: Vec<()> = vec![];
        alt(v.iter().map(|()| owned_context(rule_to_parser())).collect::<DynamicAlt<_>>())(input)
    }
}

fn main() {}
