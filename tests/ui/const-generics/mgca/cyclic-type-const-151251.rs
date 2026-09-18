//@ needs-rustc-debug-assertions

#![feature(min_generic_const_args)]
#![feature(generic_const_exprs)]
#![expect(incomplete_features)]

const A: u8 = core::direct_const_arg!(A);
//~^ ERROR cycle detected when computing the type-level value for `A`

fn main() {}
