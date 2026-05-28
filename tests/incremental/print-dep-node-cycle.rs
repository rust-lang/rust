//@ compile-flags: -Z query-dep-graph
//@ revisions: rpass1

// Exercises a debug-assertions-only query cycle that when printing a valtree const in
// a dep node's debug representation, we end up invoking a query that also has a valtree
// const in its dep node's debug representation, which leads to a cycle (and ICE, since
// deps are not tracked when printing dep nodes' debug representations).

#![feature(adt_const_params)]

use std::marker::ConstParamTy;

#[derive(Debug, ConstParamTy, PartialEq, Eq)]
enum Foo {
    A1,
}

#[inline(never)]
fn hello<const F: Foo>() {
    println!("{:#?}", F);
}

fn main() {
    hello::<{ Foo::A1 }>();
}
