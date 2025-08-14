// Test that we are able to introduce a negative constraint that
// `MyType: !MyTrait` along with other "fundamental" wrappers.

//@ aux-build:coherence_copy_like_lib.rs
#![allow(dead_code)]

extern crate coherence_copy_like_lib as lib;

struct MyType { x: i32 }

// These are all legal because they are all fundamental types:

// Tuples are not fundamental, so this is not a local impl.
impl lib::MyCopy for (MyType,) { }
//~^ ERROR E0117


fn main() { }
