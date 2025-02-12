//! Regression test for <https://github.com/rust-lang/rust/issues/133606>.
//!
//! In rare cases, all of a function's coverage spans are discarded at a late
//! stage during codegen. When that happens, the subsequent code needs to take
//! special care to avoid emitting coverage metadata that would cause `llvm-cov`
//! to fail with a fatal error.
//!
//! We currently don't know of a concise way to reproduce that scenario with
//! ordinary Rust source code, so instead we set a special testing-only flag to
//! force it to occur.

//@ edition: 2021
//@ compile-flags: -Zcoverage-options=discard-all-spans-in-codegen

// The `llvm-cov` tool will complain if the test binary ends up having no
// coverage metadata at all. To prevent that, we also link to instrumented
// code in an auxiliary crate that doesn't have the special flag set.

//@ aux-build: discard_all_helper.rs
extern crate discard_all_helper;

fn main() {
    discard_all_helper::external_function();
}
