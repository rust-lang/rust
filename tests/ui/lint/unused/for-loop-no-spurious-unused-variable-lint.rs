//! Regression test for <https://github.com/rust-lang/rust/issues/30371>.
//! This used to emit unused variable warning.
//@ run-pass

#![allow(unreachable_code)]
#![allow(for_loops_over_fallibles)]
#![deny(unused_variables)]

fn main() {
    for _ in match return () {
        () => Some(0),
    } {}
}
