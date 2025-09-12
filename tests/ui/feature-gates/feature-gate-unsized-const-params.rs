#![feature(adt_const_params)]

struct Bar(u8);

struct Foo<const N: Bar>;
//~^ ERROR: `[u8]` is forbidden as the type of a const generic parameter

fn main() {}
