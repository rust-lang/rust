//~ ERROR overflow evaluating the requirement `Self well-formed`
//~| ERROR overflow evaluating the requirement `Self: Trait`

// This is a non-regression test for issue #115351, where a recursion limit of 0 caused an ICE.
//@ compile-flags: -Znext-solver --crate-type=lib
//@ check-fail

#![recursion_limit = "0"]
trait Trait {}
impl Trait for u32 {}
//~^ ERROR overflow evaluating the requirement `u32: Trait`
//~| ERROR overflow evaluating the requirement `u32 well-formed`
