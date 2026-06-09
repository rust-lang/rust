#![feature(coverage_attribute)]
//@ edition: 2021
//@ reference: attributes.coverage.syntax

// Demonstrates the diagnostics produced when using the syntax `#[coverage]`,
// which should not be allowed.
//
// The syntax is tested both in places that can have a coverage attribute,
// and in places that cannot have a coverage attribute, to demonstrate the
// interaction between multiple errors.

#[coverage]
//~^ ERROR malformed `coverage` attribute input
mod my_mod {}

mod my_mod_inner {
    #![coverage]
    //~^ ERROR malformed `coverage` attribute input
}

#[coverage]
//~^ ERROR malformed `coverage` attribute input
//~| ERROR attribute cannot be used on
struct MyStruct;

#[coverage]
//~^ ERROR malformed `coverage` attribute input
impl MyStruct {
    #[coverage]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute cannot be used on
    const X: u32 = 7;
}

#[coverage]
//~^ ERROR malformed `coverage` attribute input
//~| ERROR attribute cannot be used on
trait MyTrait {
    #[coverage]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute cannot be used on
    const X: u32;

    #[coverage]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute cannot be used on
    type T;
}

#[coverage]
//~^ ERROR malformed `coverage` attribute input
impl MyTrait for MyStruct {
    #[coverage]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute cannot be used on
    const X: u32 = 8;

    #[coverage]
    //~^ ERROR malformed `coverage` attribute input
    //~| ERROR attribute cannot be used on
    type T = ();
}

#[coverage]
//~^ ERROR malformed `coverage` attribute input
fn main() {}
