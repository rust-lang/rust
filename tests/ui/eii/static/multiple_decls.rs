#![feature(extern_item_impls)]

const A: () = ();
#[eii(A)]
static A: u64;
//~^ ERROR the name `A` is defined multiple times
//~| ERROR expected function or static, found constant `A`

#[A]
static A_IMPL: u64 = 5;

fn main() {}
