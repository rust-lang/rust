// Check that we error when `'_` appears as the name of a lifetime parameter.
//
// Regression test for #52098.

// revisions: Rust2015 Rust2018
//[Rust2018] edition:2018

struct IceCube<'a> {
    v: Vec<&'a char>
}

impl<'_> IceCube<'_> {}
//[Rust2015]~^ ERROR
//[Rust2015]~| ERROR
//[Rust2018]~^^^ ERROR

struct Struct<'_> {
//[Rust2015]~^ ERROR
//[Rust2018]~^^ ERROR
    v: Vec<&'static char>
}

enum Enum<'_> {
//[Rust2015]~^ ERROR
//[Rust2018]~^^ ERROR
    Variant
}

union Union<'_> {
//[Rust2015]~^ ERROR
//[Rust2018]~^^ ERROR
    a: u32
}

trait Trait<'_> {
//[Rust2015]~^ ERROR
//[Rust2018]~^^ ERROR
}

fn foo<'_>() {
    //[Rust2015]~^ ERROR
    //[Rust2018]~^^ ERROR
}

fn main() {}
