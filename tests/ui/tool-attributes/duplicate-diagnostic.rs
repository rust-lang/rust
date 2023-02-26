// aux-build: p1.rs
// aux-build: p2.rs

// error-pattern: duplicate diagnostic item in crate `p2`
// error-pattern: note: the diagnostic item is first defined in crate `p1`

#![feature(rustc_attrs)]
extern crate p1;
extern crate p2;

#[rustc_diagnostic_item = "Foo"]
pub struct Foo {} //~ ERROR duplicate diagnostic item found
fn main() {}
