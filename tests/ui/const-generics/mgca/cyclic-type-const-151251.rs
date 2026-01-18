//@ needs-rustc-debug-assertions

#![feature(min_generic_const_args)]
#![feature(generic_const_exprs)]
#![expect(incomplete_features)]

#[type_const]
const A: u8 = A;
//~^ ERROR overflow normalizing the unevaluated constant `A`

fn main() {}
