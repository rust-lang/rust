#![feature(fn_delegation)]
#![allow(incomplete_features)]

fn a() {}

reuse a as b { #![rustc_dummy] self } //~ ERROR an inner attribute is not permitted in this context

fn main() {}
