//@ check-pass
//! Regression test for ICE in `rustc_hir_typeck::expr_use_visitor` on nesting a slice pattern
//! inside a deref pattern inside a closure: rust-lang/rust#125059

#![feature(deref_patterns)]
#![allow(incomplete_features, unused)]

fn simple_vec(vec: Vec<u32>) -> u32 {
   (|| match Vec::<u32>::new() {
        deref!([]) => 100,
        _ => 2000,
    })()
}

fn main() {}
