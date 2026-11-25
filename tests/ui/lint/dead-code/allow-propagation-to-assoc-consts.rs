#![feature(gca_min_const_items)]
#![allow(incomplete_features)]
#![deny(unfulfilled_lint_expectations)]

trait Tr {
    #[expect(dead_code)]
    #[rustc_always_gca]
    const C: usize;
}

#[allow(dead_code)]
fn foo<T: Tr<C = 0>>() {}

fn main() {}
