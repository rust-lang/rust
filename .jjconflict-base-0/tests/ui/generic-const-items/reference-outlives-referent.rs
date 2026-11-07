// Test that we catch that the reference outlives the referent and we
// successfully emit a diagnostic. Regression test for issue #114714.

#![feature(generic_const_items)]
#![allow(incomplete_features)]

const Q<'a, 'b>: &'a &'b () = &&(); //~ ERROR reference has a longer lifetime than the data it references

fn main() {}
