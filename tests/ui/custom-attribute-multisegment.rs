// Unresolved multi-segment attributes are not treated as custom.

mod existent {}

#[existent::nonexistent] //~ ERROR cannot find macro `nonexistent` in module `existent`
fn main() {}
