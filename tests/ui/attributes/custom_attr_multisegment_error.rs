//! Unresolved multi-segment attributes are not treated as custom.

mod existent {}

#[existent::nonexistent] //~ ERROR cannot find `nonexistent` in `existent`
fn main() {}
