//! Regression test for <https://github.com/rust-lang/rust/issues/5315>.
//! Test calling tuple struct constructor doesn't cause segfault.
//@ run-pass

struct A(#[allow(dead_code)] bool);

pub fn main() {
    let f = A;
    f(true);
}
