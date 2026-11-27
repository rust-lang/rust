// This test makes sure that we detect changed feature gates.

//@ revisions: rpass1 bfail2
//@ compile-flags: -Z query-dep-graph

#![cfg_attr(rpass1, feature(decl_macro))]

fn main() {}

macro foo() {}
//[bfail2]~^ ERROR `macro` is experimental [E0658]
