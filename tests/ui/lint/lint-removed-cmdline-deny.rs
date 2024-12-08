// The raw_pointer_derived lint warns about its removal
// cc #30346

//@ compile-flags:-D renamed-and-removed-lints -D raw_pointer_derive

//@ error-pattern:lint `raw_pointer_derive` has been removed
//@ error-pattern:requested on the command line with `-D raw_pointer_derive`
//@ error-pattern:requested on the command line with `-D renamed-and-removed-lints`

#![warn(unused)]

#[deny(warnings)]
fn main() { let unused = (); }
