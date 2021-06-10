// --extern-location with a raw reference

// aux-crate:bar=bar.rs
// compile-flags:--extern-location bar=raw -Z unstable-options

#![warn(unused_crate_dependencies)]

fn main() {}
