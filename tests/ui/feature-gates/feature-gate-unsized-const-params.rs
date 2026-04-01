struct Foo<const N: [u8]>;
//~^ ERROR: `[u8]` is forbidden as the type of a const generic parameter
//~| HELP: add `#![feature(adt_const_params)]` to the crate
//~| HELP: add `#![feature(unsized_const_params)]` to the crate

fn main() {}
