// The raw_pointer_derived lint warns about its removal
// cc #30346

// compile-flags:-D raw_pointer_derive

// error-pattern:lint `raw_pointer_derive` has been removed
// error-pattern:requested on the command line with `-D raw_pointer_derive`

#![warn(unused)]

#[deny(warnings)]
fn main() { let unused = (); }
