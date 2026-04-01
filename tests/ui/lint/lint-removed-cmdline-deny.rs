// The raw_pointer_derived lint warns about its removal
// cc #30346

//@ compile-flags:-D renamed-and-removed-lints -D raw_pointer_derive
//@ dont-require-annotations: NOTE

#![warn(unused)]

#[deny(warnings)]
fn main() { let unused = (); } //~ ERROR unused variable: `unused`

//~? ERROR lint `raw_pointer_derive` has been removed: using derive with raw pointers is ok
//~? ERROR lint `raw_pointer_derive` has been removed: using derive with raw pointers is ok
//~? ERROR lint `raw_pointer_derive` has been removed: using derive with raw pointers is ok
//~? NOTE requested on the command line with `-D raw_pointer_derive`
//~? NOTE requested on the command line with `-D renamed-and-removed-lints`
