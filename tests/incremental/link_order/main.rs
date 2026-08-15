//@ aux-build:my_lib.rs
//@ revisions: bfail1 bfail2
//@ compile-flags:-Z query-dep-graph
//@ ignore-backends: gcc

// Tests that re-ordering the `-l` arguments used
// when compiling an external dependency does not lead to
// an 'unstable fingerprint' error.

extern crate my_lib;

fn main() {}

//~? ERROR linking with
