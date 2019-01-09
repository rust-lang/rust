// Test that we are able to introduce a negative constraint that
// `MyType: !MyTrait` along with other "fundamental" wrappers.

// aux-build:coherence_copy_like_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_copy_like_lib as lib;

struct MyType { x: i32 }

trait MyTrait { fn foo() {} }
impl<T: lib::MyCopy> MyTrait for T { }

// Tuples are not fundamental, therefore this would require that
//
//     (MyType,): !MyTrait
//
// which we cannot approve.
impl MyTrait for (MyType,) { }
//[old]~^ ERROR E0119
//[re]~^^ ERROR E0119

fn main() { }
