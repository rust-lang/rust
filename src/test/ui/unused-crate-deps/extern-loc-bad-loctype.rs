// --extern-location with bad location type

// aux-crate:bar=bar.rs
// compile-flags:--extern-location bar=badloc:in-the-test-file

#![warn(unused_crate_dependencies)]

fn main() {}
