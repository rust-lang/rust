//@ proc-macro: attributes-included.rs
//@ check-pass

#![warn(unused)]

extern crate attributes_included;

use attributes_included::*;

#[bar]
#[inline]
/// doc
#[foo]
#[inline]
/// doc
fn foo() {
    let a: i32 = "foo"; //~ WARN: unused variable
}

fn main() {
    foo()
}
