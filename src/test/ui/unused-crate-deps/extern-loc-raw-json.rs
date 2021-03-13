// --extern-location with a raw reference

// check-pass
// aux-crate:bar=bar.rs
// compile-flags:--extern-location bar=raw:in-the-test-file --error-format json

#![warn(unused_crate_dependencies)]
//~^ WARNING external crate `bar` unused in

fn main() {}
