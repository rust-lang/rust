// Checks that a const param cannot be stored in a struct.
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct S<const C: u8>(C); //~ ERROR expected type, found const parameter

fn main() {}
