//! Test that we can match on enum constants across crates.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/24106>.


//@ run-pass
//@ aux-build:cross-crate-enum-pattern.rs

extern crate cross_crate_enum_pattern;

fn main() {
    cross_crate_enum_pattern::go::<()>();
}
