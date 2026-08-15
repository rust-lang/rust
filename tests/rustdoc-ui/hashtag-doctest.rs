// This test ensures that `##` are not emitting a warning when generating
// docs with the 2024 edition (or any edition).
// Regression test for <https://github.com/rust-lang/rust/issues/136899>.

//@ check-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

#![deny(warnings)]

//! Test
//!
//! ```
//! ##[allow(dead_code)]
//! println!("hello world");
//! ```
