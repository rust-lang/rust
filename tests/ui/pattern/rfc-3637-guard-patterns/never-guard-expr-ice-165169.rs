//@ check-pass
//! Regression test for <https://github.com/rust-lang/rust/issues/165169>.
//! Make sure a never-typed guard expression in a guard pattern doesn't ICE.

#![feature(guard_patterns)]
#![expect(incomplete_features)]
#![allow(unused_parens, unused_variables)]

fn main() {
    let x: !;
    match 1 {
        ((y if x)) => {}
    }
}
