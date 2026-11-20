// `unused_crate_dependencies` is not currently in the `unused` group
// due to false positives from Cargo.

//@ check-pass
//@ aux-crate:bar=bar.rs

#![deny(unused)]

fn main() {}
