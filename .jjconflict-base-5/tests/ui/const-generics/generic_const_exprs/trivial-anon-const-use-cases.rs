//! Regression test for <https://github.com/rust-lang/rust/issues/79429>.

//@ revisions: full min
#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

struct X<const S: usize>;

impl<const S: usize> X<S> {
    const LEN: usize = S + 1; // `S + 1` is a valid const expression in this context.
}

struct Y<const S: usize> {
    stuff: [u8; { S + 1 }], // `S + 1` is NOT a valid const expression in this context.
    //[min]~^ ERROR generic parameters may not be used in const operations
    //[full]~^^ ERROR unconstrained generic constant
}

fn main() {}
