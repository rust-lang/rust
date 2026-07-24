//@ compile-flags: --test --crate-type=lib,bin
// A library crate compiled with `--test`.

#![deny(dead_code)]
#![deny(dead_code_pub_in_binary)]

pub fn unused_pub_fn() {} // Should NOT error because this is a library compiled for testing

fn unused_priv_fn() {} //~ ERROR function `unused_priv_fn` is never used
