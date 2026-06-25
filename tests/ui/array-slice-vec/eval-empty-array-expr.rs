//! Regression test for <https://github.com/rust-lang/rust/issues/23354>.
//! Check expr in [expr; N] is always being evaluated.
//@ run-fail
//@ error-pattern:panic evaluated
//@ needs-subprocess

#[allow(unused_variables)]
fn main() {
    let x = [panic!("panic evaluated"); 0];
}
