// No warnings about removed lint when
// allow(renamed_and_removed_lints)

#![allow(renamed_and_removed_lints)]

#[deny(raw_pointer_derive)]
#[deny(unused_variables)]
fn main() { let unused = (); } //~ ERROR unused
