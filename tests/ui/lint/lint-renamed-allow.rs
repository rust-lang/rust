// No warnings about renamed lint when
// allow(renamed_and_removed_lints)

#![allow(renamed_and_removed_lints)]

#[deny(single_use_lifetime)]
#[deny(unused)]
fn main() { let unused = (); } //~ ERROR unused
