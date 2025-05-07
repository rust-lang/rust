//@ check-pass
//@ compile-flags: -Znext-solver --crate-type=lib

// This is a non-regression test for issue #115351, where a recursion limit of 0 caused an ICE.

#![recursion_limit = "0"]
trait Trait {}
impl Trait for u32 {}
