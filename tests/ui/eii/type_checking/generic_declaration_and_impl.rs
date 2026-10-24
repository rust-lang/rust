//@ check-fail
// Check that comparing an EII implementation against a generic declaration
// (already rejected by E0044) does not ICE in error reporting.
// Regression test for the ICE when note_and_explain_type_err tries to look up
// declaration parameters in external_impl's empty generics.
#![feature(extern_item_impls)]

#[eii(foo)]
fn bar<T>(x: T) where T: FnOnce() -> u64 { x() }
//~^ ERROR externally implementable items may not have type parameters
//~| ERROR mismatched types

#[foo]
fn y(a: u64) -> u64 {
    a
}

fn main() {}
