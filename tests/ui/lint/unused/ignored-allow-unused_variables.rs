//! Regression test for <https://github.com/rust-lang/rust/issues/43910>.
//! This used to emit unused variable lint even if it is explicitly allowed.
//@ run-pass

#![deny(unused_variables)]

fn main() {
    #[allow(unused_variables)]
    let x = 12;
}
