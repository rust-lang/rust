// Test that lint `explicit_outlives_requirements` can also trigger on GCIs.

//@ check-pass
//@ run-rustfix

#![feature(generic_const_items)]
#![warn(explicit_outlives_requirements)]
#![allow(dead_code)]

const C<'a, T: 'a>: Option<&'a T> = None; //~ WARN [explicit_outlives_requirements]

const K<'a, T>: Option<&'a T> = None where T: 'a; //~ WARN [explicit_outlives_requirements]

fn main() {}
