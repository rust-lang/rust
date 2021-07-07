// --extern-location with no type

// aux-crate:bar=bar.rs
// compile-flags:--extern-location bar=missing-loc-type -Z unstable-options

#![warn(unused_crate_dependencies)]

fn main() {}
