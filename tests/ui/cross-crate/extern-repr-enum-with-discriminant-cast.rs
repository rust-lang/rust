//@ run-pass
#![allow(dead_code)]
//@ aux-build:extern-repr-enum-with-discriminant-cast.rs
//! Regression test for https://github.com/rust-lang/rust/issues/42007
//! This test exposes a bug that happens because the Session LintStore is emptied when linting
//! The LintStore is emptied because the checker wants ownership as it wants to
//! mutate the pass objects and lint levels.
//! The fix was not taking ownership of the entire store just the lint levels and pass objects.
extern crate extern_repr_enum_with_discriminant_cast as issue_42007_s;

enum I {
    E(issue_42007_s::E),
}

fn main() {}
