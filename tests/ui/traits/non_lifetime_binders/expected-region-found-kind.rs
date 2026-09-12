//! Regression test for <https://github.com/rust-lang/rust/issues/155802>.
//@ check-fail

#![feature(non_lifetime_binders)]
trait E<'e> {
    type As;
}

trait F<'a>: for<F> E<'a> + for<'e> E<'e> {}
//~^ ERROR type annotations needed: cannot satisfy `Self: E<'a>` [E0283]

struct G<'a, T>
where
    T: F<'a, As: E<'a>>,
    //~^ ERROR type annotations needed: cannot satisfy `T: E<'a>` [E0283]
    //~| ERROR ambiguous associated type `As` in bounds of `F` [E0221]
{
    x: &'a T,
}

fn main() {}
