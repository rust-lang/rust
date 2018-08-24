// Unresolved multi-segment attributes are not treated as custom.

#![feature(custom_attribute)]

mod existent {}

#[existent::nonexistent] //~ ERROR failed to resolve. Could not find `nonexistent` in `existent`
fn main() {}
