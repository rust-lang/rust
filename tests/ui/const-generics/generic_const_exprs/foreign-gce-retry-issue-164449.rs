// Regression test for #164449.
// Cross-crate GCE must not ICE in the next solver.

//@ edition: 2024
//@ compile-flags: -Znext-solver=globally --crate-type=lib
//@ aux-crate: gce_dependency=foreign-gce-retry-164449.rs
//@ check-fail

use gce_dependency::S;

pub fn f() {
    let _ = S::<8>::new();
}

// Errors are reported from the dependency.
//~? ERROR type annotations needed [E0284]
//~? ERROR type annotations needed [E0284]
