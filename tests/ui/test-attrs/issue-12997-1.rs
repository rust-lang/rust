// compile-flags: --test

//! Test that makes sure wrongly-typed bench functions aren't ignored

#![feature(test)]

#[bench]
fn foo() { } //~ ERROR functions used as benches

#[bench]
fn bar(x: isize, y: isize) { } //~ ERROR functions used as benches
