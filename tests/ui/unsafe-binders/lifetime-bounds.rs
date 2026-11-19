//@ compile-flags: -Znext-solver
//@ check-fail

// Lifetimes bound by an unsafe binder can't have bounds.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

type Static = unsafe<'a: 'static> &'a (); //~ ERROR bounds cannot be used in this context

fn f(_: unsafe<'a, 'b: 'a> &'a &'b ()) {} //~ ERROR bounds cannot be used in this context

fn main() {}
