#![feature(coverage_attribute)]
//@ edition: 2021

// Demonstrates the diagnostics produced when using the syntax
// `#[coverage = "off"]`, which should not be allowed.
//
// The syntax is tested both in places that can have a coverage attribute,
// and in places that cannot have a coverage attribute, to demonstrate the
// interaction between multiple errors.

#[coverage = "off"]
//~^ ERROR malformed `coverage` attribute input
mod my_mod {}

mod my_mod_inner {
    #![coverage = "off"]
    //~^ ERROR malformed `coverage` attribute input
}

#[coverage = "off"]
//~^ ERROR malformed `coverage` attribute input
//~| ERROR attribute should be applied to a function definition or closure
struct MyStruct;

#[coverage = "off"]
//~^ ERROR malformed `coverage` attribute input
impl MyStruct {
    #[coverage = "off"]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute should be applied to a function definition or closure
    const X: u32 = 7;
}

#[coverage = "off"]
//~^ ERROR malformed `coverage` attribute input
//~| ERROR attribute should be applied to a function definition or closure
trait MyTrait {
    #[coverage = "off"]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute should be applied to a function definition or closure
    const X: u32;

    #[coverage = "off"]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute should be applied to a function definition or closure
    type T;
}

#[coverage = "off"]
//~^ ERROR malformed `coverage` attribute input
impl MyTrait for MyStruct {
    #[coverage = "off"]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute should be applied to a function definition or closure
    const X: u32 = 8;

    #[coverage = "off"]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute should be applied to a function definition or closure
    type T = ();
}

#[coverage = "off"]
//~^ ERROR malformed `coverage` attribute input
fn main() {}
