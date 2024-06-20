#![feature(coverage_attribute)]
//@ edition: 2021

// Demonstrates the diagnostics produced when using the syntax
// `#[coverage = "off"]`, which should not be allowed.
//
// The syntax is tested both in places that can have a coverage attribute,
// and in places that cannot have a coverage attribute, to demonstrate the
// interaction between multiple errors.

// FIXME(#126658): The error messages for using this syntax are inconsistent
// with the error message in other cases. They also sometimes appear together
// with other errors, and they suggest using the incorrect `#[coverage]` syntax.

#[coverage = "off"] //~ ERROR malformed `coverage` attribute input
mod my_mod {}

mod my_mod_inner {
    #![coverage = "off"] //~ ERROR malformed `coverage` attribute input
}

#[coverage = "off"]
//~^ ERROR `#[coverage]` must be applied to coverable code
//~| ERROR malformed `coverage` attribute input
struct MyStruct;

#[coverage = "off"] //~ ERROR malformed `coverage` attribute input
impl MyStruct {
    #[coverage = "off"]
    //~^ ERROR `#[coverage]` must be applied to coverable code
    //~| ERROR malformed `coverage` attribute input
    const X: u32 = 7;
}

#[coverage = "off"] //~ ERROR malformed `coverage` attribute input
trait MyTrait {
    #[coverage = "off"]
    //~^ ERROR `#[coverage]` must be applied to coverable code
    //~| ERROR malformed `coverage` attribute input
    const X: u32;

    #[coverage = "off"]
    //~^ ERROR `#[coverage]` must be applied to coverable code
    //~| ERROR malformed `coverage` attribute input
    type T;
}

#[coverage = "off"] //~ ERROR malformed `coverage` attribute input
impl MyTrait for MyStruct {
    #[coverage = "off"]
    //~^ ERROR `#[coverage]` must be applied to coverable code
    //~| ERROR malformed `coverage` attribute input
    const X: u32 = 8;

    #[coverage = "off"]
    //~^ ERROR `#[coverage]` must be applied to coverable code
    //~| ERROR malformed `coverage` attribute input
    type T = ();
}

#[coverage = "off"]
//~^ ERROR expected `coverage(off)` or `coverage(on)`
//~| ERROR malformed `coverage` attribute input
fn main() {}
