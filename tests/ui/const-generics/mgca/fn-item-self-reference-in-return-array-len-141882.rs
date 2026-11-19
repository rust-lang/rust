//! Regression test for <https://github.com/rust-lang/rust/issues/141882>.
#![feature(gca_min_const_items, gca_macroless_args)]

use std::gca;

fn combinator<T, const S: usize>() -> [T; combinator] {}
//~^ ERROR function items cannot be used as const args

fn combinator_macro<T, const S: usize>() -> [T; gca!(combinator_macro)] {}
//~^ ERROR function items cannot be used as const args

fn main() {}
