//! Short backtraces should still be emitted from stripped binaries.
//! Regression test for https://github.com/rust-lang/rust/issues/147846
//
//@ compile-flags: -Cstrip=symbols
//@ exec-env: RUST_BACKTRACE=1
//@ run-fail
//@ check-run-results
//
//  Name mangling scheme differences
//@ normalize-stderr: "begin_panic::<&str>" -> "begin_panic"
//
//  macOS with `rust.debuginfo-level = "line-tables-only"` (#133997)
//@ normalize-stderr: " begin_panic<&str>" -> " std::panicking::begin_panic"
//
//  debuginfo
//@ normalize-stderr: "\n +at [^\n]+" -> ""

fn main() { panic!(); }
