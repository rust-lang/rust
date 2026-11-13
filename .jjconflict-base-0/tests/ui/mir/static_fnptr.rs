//! Verify that we correctly handle fn pointer provenance in MIR optimizations.
//! By asking to inline `static_fnptr::bar`, we have two copies of `static_fnptr::foo`, one in the
//! auxiliary crate and one in the local crate CGU.
//! `baz` must only consider the versions from upstream crate, and not try to compare with the
//! address of the CGU-local copy.
//! Related issue: #123670

//@ run-pass
//@ compile-flags:-Cno-prepopulate-passes -Copt-level=0
//@ aux-build:static_fnptr.rs

extern crate static_fnptr;
use static_fnptr::{ADDR, bar};

fn baz() -> bool {
    bar(ADDR)
}

fn main() {
    assert!(baz())
}
