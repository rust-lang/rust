//@ aux-build:my_lib.rs
//@ revisions:cfail1 cfail2
//@ compile-flags:-Z query-dep-graph

// Tests that re-ordering the `-l` arguments used
// when compiling an external dependency does not lead to
// an 'unstable fingerprint' error.

extern crate my_lib;

fn main() {}

//~? ERROR linking with
