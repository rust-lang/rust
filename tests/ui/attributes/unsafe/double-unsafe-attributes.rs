#![feature(unsafe_attributes)]

#[unsafe(unsafe(no_mangle))]
//~^ ERROR expected identifier, found keyword `unsafe`
//~| ERROR cannot find attribute `r#unsafe`
//~| ERROR `r#unsafe` is not an unsafe attribute
fn a() {}

fn main() {}
