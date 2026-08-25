//! Regression test for <https://github.com/rust-lang/rust/issues/117923>.
//@ check-pass
#![feature(min_generic_const_args, macroless_generic_const_args)]
#![allow(incomplete_features, dead_code)]

trait Trait {
    type const CT: usize;
}

struct Type<const N: usize> {
    field: [u8; N],
}

impl<const N: usize> Trait for Type<N> {
    type const CT: usize = N;
}

fn func<const N: usize>() -> impl Trait<CT = { <Type<N> as Trait>::CT }> {
    Type { field: [0; N] }
}

fn main() {}
