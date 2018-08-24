extern crate std as _; //~ ERROR renaming extern crates with `_` is unstable
use std::vec as _; //~ ERROR renaming imports with `_` is unstable

fn main() {}
