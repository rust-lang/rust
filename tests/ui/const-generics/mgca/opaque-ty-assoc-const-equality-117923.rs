//! Regression test for <https://github.com/rust-lang/rust/issues/117923>.
//@ check-pass
#![feature(gca_min_const_items, gca_macroless_args)]
#![allow(incomplete_features, dead_code)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const CT: usize;
}

struct Type<const N: usize> {
    field: [u8; N],
}

impl<const N: usize> Trait for Type<N> {
    const CT: usize = gca!(N);
}

fn func<const N: usize>() -> impl Trait<CT = { <Type<N> as Trait>::CT }> {
    Type { field: [0; N] }
}

fn main() {}
