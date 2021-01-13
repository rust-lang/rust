// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub const fn func_name<const X: *const u32>() {}
//~^ ERROR using raw pointers

fn main() {}
