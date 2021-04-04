// [full] check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn test<const N: [u8; 1 + 2]>() {}
//[min]~^ ERROR `[u8; _]` is forbidden as the type of a const generic parameter

struct Foo<const N: [u8; 1 + 2]>;
//[min]~^ ERROR `[u8; _]` is forbidden as the type of a const generic parameter

fn main() {}
