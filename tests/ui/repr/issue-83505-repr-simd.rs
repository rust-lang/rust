// Regression test for the ICE described in #83505.

#![crate_type="lib"]

#[repr(simd)]
//~^ ERROR: attribute should be applied to a struct [E0517]
//~| ERROR: unsupported representation for zero-variant enum [E0084]
enum Es {}
static CLs: Es;
//~^ ERROR: free static item without body
