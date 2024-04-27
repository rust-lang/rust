// Regression test for the ICE described in #83475.

#![crate_type="lib"]

#![feature(cmse_nonsecure_entry)]
#[cmse_nonsecure_entry]
//~^ ERROR: attribute should be applied to a function definition
struct XEmpty2;
//~^ NOTE: not a function definition
