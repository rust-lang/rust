// Test that we are able to introduce a negative constraint that
// `MyType: !MyTrait` along with other "fundamental" wrappers.

// check-pass
// aux-build:coherence_copy_like_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_copy_like_lib as lib;

struct MyType { x: i32 }

trait MyTrait { fn foo() {} }
impl<T: lib::MyCopy> MyTrait for T { }

// `MyFundamentalStruct` is declared fundamental, so we can test that
//
//    MyFundamentalStruct<&MyTrait>: !MyTrait
//
// Huzzah.
impl<'a> MyTrait for lib::MyFundamentalStruct<&'a MyType> { }

fn main() { }
