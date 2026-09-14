#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

const A: u8 = core::direct_const_arg!(A);
//~^ ERROR: cycle detected when computing the type-level value for `A` [E0391]

fn main() {}
