//@ compile-flags: -Znext-solver
//@ check-pass
//@ known-bug: #130516

// Lifetimes bound by an unsafe binder can't have bounds.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

type Static = unsafe<'a: 'static> &'a ();

fn f(_: unsafe<'a, 'b: 'a> &'a &'b ()) {}

fn main() {}
