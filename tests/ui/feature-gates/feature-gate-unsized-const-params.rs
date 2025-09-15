#![feature(adt_const_params)]

struct Bar(u8);

struct Foo<const N: Bar>;
//~^ ERROR: `Bar` must implement `ConstParamTy` to be used as the type of a const generic parameter

fn main() {}
