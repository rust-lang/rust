// Check that we error when `'_` appears as the name of a lifetime parameter.
//
// Regression test for #52098.

// revisions: Rust2015 Rust2018
//[Rust2018] edition:2018

struct IceCube<'a> {
    v: Vec<&'a char>
}

impl<'_> IceCube<'_> {}
//[Rust2015]~^ ERROR `'_` cannot be used here
//[Rust2015]~| ERROR missing lifetime specifier
//[Rust2018]~^^^ ERROR `'_` cannot be used here

struct Struct<'_> {
//[Rust2015]~^ ERROR `'_` cannot be used here
//[Rust2018]~^^ ERROR `'_` cannot be used here
    v: Vec<&'static char>
}

enum Enum<'_> {
//[Rust2015]~^ ERROR `'_` cannot be used here
//[Rust2018]~^^ ERROR `'_` cannot be used here
    Variant
}

union Union<'_> {
//[Rust2015]~^ ERROR `'_` cannot be used here
//[Rust2018]~^^ ERROR `'_` cannot be used here
    a: u32
}

trait Trait<'_> {
//[Rust2015]~^ ERROR `'_` cannot be used here
//[Rust2018]~^^ ERROR `'_` cannot be used here
}

fn foo<'_>() {
    //[Rust2015]~^ ERROR `'_` cannot be used here
    //[Rust2018]~^^ ERROR `'_` cannot be used here
}

fn main() {}
