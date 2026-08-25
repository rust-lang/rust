//! Regression test for <https://github.com/rust-lang/rust/issues/23898>.
//! Test enum discriminants are considered consts and can be used as a paths in const expressions.
//!
//! This test was used to demonstrate <https://github.com/rust-lang/rust/issues/5873>.
//@ run-pass

#![allow(unused_parens)]
#![allow(non_camel_case_types)]

enum State { ST_NULL, ST_WHITESPACE }

fn main() {
    [State::ST_NULL; (State::ST_WHITESPACE as usize)];
}
