// --extern-location with a raw reference

// aux-crate:bar=bar.rs
// compile-flags:--extern-location bar -Zunstable-options

#![warn(unused_crate_dependencies)]

fn main() {}
