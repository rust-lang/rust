// Checks that a const param cannot be stored in a struct.
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct S<const C: u8>(C); //~ ERROR expected type, found const parameter

fn main() {}
