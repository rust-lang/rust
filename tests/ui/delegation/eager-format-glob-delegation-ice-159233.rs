//@ edition: 2024

#![feature(fn_delegation)]

static REGEX: () = format!(|| { reuse impl Trait for S; });
//~^ ERROR format argument must be a string literal
//~| ERROR cannot find type `Trait` in this scope
//~| ERROR mismatched types

fn main() {}
