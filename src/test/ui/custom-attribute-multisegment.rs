// Unresolved multi-segment attributes are not treated as custom.

mod existent {}

#[existent::nonexistent] //~ ERROR failed to resolve: could not find `nonexistent` in `existent`
fn main() {}
