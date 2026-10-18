//@ check-pass
//@ compile-flags: --test

// Regression test for #159078: a library's unit tests are compiled by cargo with `--test`
// and no `--crate-type`. Unused `pub` items must not be reported there.
#![deny(dead_code_pub_in_binary)]

pub fn unused_pub_fn() {}
