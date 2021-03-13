// Default extern location from name and path if one isn't specified

// check-pass
// aux-crate:bar=bar.rs
// compile-flags:--error-format json

#![warn(unused_crate_dependencies)]
//~^ WARNING external crate `bar` unused in

fn main() {}
