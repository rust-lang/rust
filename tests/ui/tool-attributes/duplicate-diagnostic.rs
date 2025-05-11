//@ aux-build: p1.rs
//@ aux-build: p2.rs

#![feature(rustc_attrs)]
extern crate p1;
extern crate p2;

#[rustc_diagnostic_item = "Foo"]
pub struct Foo {} //~ ERROR duplicate diagnostic item in crate `duplicate_diagnostic`: `Foo`
                  //~^ NOTE the diagnostic item is first defined in crate `p2`
fn main() {}

//~? ERROR duplicate diagnostic item in crate `p2`
//~? NOTE the diagnostic item is first defined in crate `p1`
