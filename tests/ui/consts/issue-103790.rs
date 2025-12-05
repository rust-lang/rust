#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct S<const S: (), const S: S = { S }>;
//~^ ERROR the name `S` is already used for a generic parameter in this item's generic parameters
//~| ERROR missing generics for struct `S`
//~| ERROR cycle detected when computing type of `S::S`
//~| ERROR `()` is forbidden as the type of a const generic parameter

fn main() {}
