//! Test the behavior of various malformed `crate_type` attributes applied to a non-crate target.
#![allow(unused_attributes)]

// No arguments
#[crate_type]           //~ ERROR malformed `crate_type` attribute input
// List/NameValue with/without strings
#[crate_type(lib)]      //~ ERROR malformed `crate_type` attribute input
#[crate_type("lib")]    //~ ERROR malformed `crate_type` attribute input
#[crate_type = lib]     //~ ERROR attribute value must be a literal
#[crate_type = "lib"]   //  OK
// Same as above but with invalid names
#[crate_type(foo)]      //~ ERROR malformed `crate_type` attribute input
#[crate_type("foo")]    //~ ERROR malformed `crate_type` attribute input
#[crate_type = foo]     //~ ERROR attribute value must be a literal
#[crate_type = "foo"]   //  OK - we don't report errors on invalid crate types here
// Non-string literals
#[crate_type(1)]        //~ ERROR malformed `crate_type` attribute input
#[crate_type = 1]       //~ ERROR malformed `crate_type` attribute input
fn main() {}
