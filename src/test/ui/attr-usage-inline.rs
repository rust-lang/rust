#![allow(dead_code)]

#[inline]
fn f() {}

#[inline] //~ ERROR: attribute should be applied to function or closure
struct S;

fn main() {}
