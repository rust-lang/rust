// [full] check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

const LEN: usize = 1024;

fn hoge<const IN: [u32; LEN]>() {}
//[min]~^ ERROR `[u32; _]` is forbidden as the type of a const generic parameter

fn main() {}
