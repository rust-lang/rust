// The raw_pointer_derived lint warns about its removal
// cc #30346

//@compile-flags:-D raw_pointer_derive

//@error-in-other-file:lint `raw_pointer_derive` has been removed
//@error-in-other-file:requested on the command line with `-D raw_pointer_derive`

#![warn(unused)]

#[deny(warnings)]
fn main() { let unused = (); }
