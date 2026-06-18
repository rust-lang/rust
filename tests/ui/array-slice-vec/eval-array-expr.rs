//! Regression test for <https://github.com/rust-lang/rust/issues/23354>.
//! Check expr in [expr; N] is always being evaluated.
//!
//! This used to trigger an LLVM assertion during compilation
//@ run-fail
//@ error-pattern:panic evaluated
//@ needs-subprocess

#[allow(unused_variables)]
fn main() {
    let x = [panic!("panic evaluated"); 2];
}
