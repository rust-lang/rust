//@ compile-flags: --test
// A normal crate compiled with `--test`.

#![deny(dead_code)]
#![deny(dead_code_pub_in_binary)]

pub fn unused_pub_fn() {} //~ ERROR function `unused_pub_fn` is never used

fn unused_priv_fn() {} //~ ERROR function `unused_priv_fn` is never used
