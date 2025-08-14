//! Check that UFCS syntax works correctly in return statements
//! without requiring workaround parentheses.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/37765>.

//@ run-pass
//@ run-rustfix

#![allow(dead_code)]
#![warn(unused_parens)]

fn with_parens<T: ToString>(arg: T) -> String {
    return (<T as ToString>::to_string(&arg)); //~ WARN unnecessary parentheses around `return` value
}

fn no_parens<T: ToString>(arg: T) -> String {
    return <T as ToString>::to_string(&arg);
}

fn main() {}
