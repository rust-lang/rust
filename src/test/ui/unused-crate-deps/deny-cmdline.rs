// Check for unused crate dep, no path

// edition:2018
// compile-flags: -Dunused-crate-dependencies
// aux-crate:bar=bar.rs

fn main() {}
//~^ ERROR external crate `bar` unused in
