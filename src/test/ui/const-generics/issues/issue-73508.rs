// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

pub const fn func_name<const X: *const u32>() {}
//[full]~^ ERROR using raw pointers
//[min]~^^ ERROR using raw pointers as const generic parameters is forbidden

fn main() {}
