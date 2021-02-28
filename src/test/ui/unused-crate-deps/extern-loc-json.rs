// --extern-location with a raw reference

// check-pass
// aux-crate:bar=bar.rs
// compile-flags:--extern-location bar=json:{"key":123,"value":{}} -Z unstable-options

#![warn(unused_crate_dependencies)]
//~^ WARNING external crate `bar` unused in

fn main() {}
