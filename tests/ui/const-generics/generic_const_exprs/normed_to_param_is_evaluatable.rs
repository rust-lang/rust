//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features, unused_braces)]

#[rustfmt::skip]
fn foo<const N: usize>() {
    bar::<{{{{{{ N }}}}}}>();
}

fn bar<const N: usize>() {}

fn main() {}
