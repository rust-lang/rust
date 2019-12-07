// aux-build:attributes-included.rs
// build-pass (FIXME(62277): could be check-pass?)

#![warn(unused)]

extern crate attributes_included;

use attributes_included::*;

#[bar]
#[inline]
/// doc
#[foo] //~ WARN unused variable: `a`
#[inline]
/// doc
fn foo() {
    let a: i32 = "foo";
}

fn main() {
    foo()
}
