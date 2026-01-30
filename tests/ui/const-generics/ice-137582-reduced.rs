#![feature(adt_const_params)]

const VALUE: [u32] = [0; 4];
//~^ ERROR the size for values of type `[u32]` cannot be known at compilation time
//~| ERROR mismatched types

struct SomeStruct<const V: [u32]> {}
//~^ ERROR use of unstable library feature `unsized_const_params`

impl SomeStruct<VALUE> {}
//~^ ERROR the size for values of type `[u32]` cannot be known at compilation time

fn main() {}
