//! Guard patterns require parentheses to disambiguate precedence
//!
//! Regression test for https://github.com/rust-lang/rust/issues/149594

//@ check-pass

#![feature(guard_patterns)]
#![expect(incomplete_features)]
#![warn(unused_parens)]

fn main() {
    let (_ if false) = ();
}
