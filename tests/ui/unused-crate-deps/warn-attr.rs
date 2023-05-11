// Check for unused crate dep, no path

// edition:2018
// check-pass
// aux-crate:bar=bar.rs

#![warn(unused_crate_dependencies)]
//~^ WARNING external crate `bar` unused in

fn main() {}
