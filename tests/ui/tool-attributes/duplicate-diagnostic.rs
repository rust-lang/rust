//@aux-build: p1.rs
//@aux-build: p2.rs

//@error-in-other-file: duplicate diagnostic item in crate `p2`
//@error-in-other-file: note: the diagnostic item is first defined in crate `p1`

#![feature(rustc_attrs)]
extern crate p1;
extern crate p2;

#[rustc_diagnostic_item = "Foo"]
pub struct Foo {} //~ ERROR duplicate diagnostic item in crate `duplicate_diagnostic`: `Foo`
fn main() {}
