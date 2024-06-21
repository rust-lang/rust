#![feature(coverage_attribute)]
//@ edition: 2021

// Demonstrates the diagnostics produced when using the syntax `#[coverage]`,
// which should not be allowed.
//
// The syntax is tested both in places that can have a coverage attribute,
// and in places that cannot have a coverage attribute, to demonstrate the
// interaction between multiple errors.

// FIXME(#126658): The error messages for using this syntax give the impression
// that it is legal, even though it should never be legal.

// FIXME(#126658): This is silently allowed, but should not be.
#[coverage]
mod my_mod {}

// FIXME(#126658): This is silently allowed, but should not be.
mod my_mod_inner {
    #![coverage]
}

#[coverage] //~ ERROR `#[coverage]` must be applied to coverable code
struct MyStruct;

// FIXME(#126658): This is silently allowed, but should not be.
#[coverage]
impl MyStruct {
    #[coverage] //~ ERROR `#[coverage]` must be applied to coverable code
    const X: u32 = 7;
}

// FIXME(#126658): This is silently allowed, but should not be.
#[coverage]
trait MyTrait {
    #[coverage] //~ ERROR `#[coverage]` must be applied to coverable code
    const X: u32;

    #[coverage] //~ ERROR `#[coverage]` must be applied to coverable code
    type T;
}

// FIXME(#126658): This is silently allowed, but should not be.
#[coverage]
impl MyTrait for MyStruct {
    #[coverage] //~ ERROR `#[coverage]` must be applied to coverable code
    const X: u32 = 8;

    #[coverage] //~ ERROR `#[coverage]` must be applied to coverable code
    type T = ();
}

#[coverage] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn main() {}
