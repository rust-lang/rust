// Test that we are able to introduce a negative constraint that
// `MyType: !MyTrait` along with other "fundamental" wrappers.

// aux-build:coherence_copy_like_lib.rs

#![feature(rustc_attrs)]
#![allow(dead_code)]

extern crate coherence_copy_like_lib as lib;

struct MyType { x: i32 }

// naturally, legal
impl lib::MyCopy for MyType { }

#[rustc_error]
fn main() { } //~ ERROR compilation successful
