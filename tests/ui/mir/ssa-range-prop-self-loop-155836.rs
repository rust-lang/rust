// Regression test for <https://github.com/rust-lang/rust/issues/155836>.
// Ensure SSA range propagation does not ICE on self-loops.
//@ compile-flags: -C opt-level=2
//@ check-pass

#![crate_type = "lib"]

pub fn trigger(b: usize) -> usize {
    while 0 != 2 {
        b % b;
    }
    b
}
