//@ [full] check-pass
//@ revisions: full min
#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

fn test<const N: [u8; 1 + 2]>() {}
//[min]~^ ERROR `[u8; 1 + 2]` is forbidden as the type of a const generic parameter

struct Foo<const N: [u8; 1 + 2]>;
//[min]~^ ERROR `[u8; 1 + 2]` is forbidden as the type of a const generic parameter

fn main() {}
