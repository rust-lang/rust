//! If we give LLVM a local file table for a function, but some of the entries
//! in that table have no associated mapping regions, then an assertion failure
//! will occur in LLVM. We therefore need to detect and skip any function that
//! would trigger that assertion.
//!
//! To test that this case is handled, even before adding code that could allow
//! it to happen organically (for expansion region support), we use a special
//! testing-only flag to force it to occur.

//@ edition: 2024
//@ compile-flags: -Zcoverage-options=inject-unused-local-file

// The `llvm-cov` tool will complain if the test binary ends up having no
// coverage metadata at all. To prevent that, we also link to instrumented
// code in an auxiliary crate that doesn't have the special flag set.

//@ aux-build: discard_all_helper.rs
extern crate discard_all_helper;

fn main() {
    discard_all_helper::external_function();
}
