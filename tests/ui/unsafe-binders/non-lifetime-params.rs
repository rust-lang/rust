//@ compile-flags: -Znext-solver
//@ check-fail

// Unsafe binders cannot bind type and const params. See #141293.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

type WithType = unsafe<T> (); //~ ERROR only lifetime parameters can be used in this context
type WithConst = unsafe<const N: usize> [u8; N]; //~ ERROR only lifetime parameters can be used in this context

fn main() {}
