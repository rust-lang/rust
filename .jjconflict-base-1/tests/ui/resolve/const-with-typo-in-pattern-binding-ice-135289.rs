// This is a non-regression test for issue 135289, where the "const with typo in pattern" diagnostic
// caused an ICE when unexpectedly pretty printing a type for unreachable arms via a macro defined
// in a dependency.

#![warn(unreachable_patterns)] // needed to reproduce the ICE described in #135289

//@ check-pass
//@ aux-build: fake_matches.rs
extern crate fake_matches;

const _A: u64 = 0;
pub fn f() -> u64 {
    0
}
fn main() {
    fake_matches::assert_matches!(f(), _non_existent);
}
