//@ run-pass
#![allow(dead_code)]
// Test that we are able to introduce a negative constraint that
// `MyType: !MyTrait` along with other "fundamental" wrappers.

//@ aux-build:coherence_copy_like_lib.rs

extern crate coherence_copy_like_lib as lib;

struct MyType { x: i32 }

trait MyTrait { }
impl<T: lib::MyCopy> MyTrait for T { }
impl MyTrait for MyType { }
impl<'a> MyTrait for &'a MyType { }
impl MyTrait for Box<MyType> { }
impl<'a> MyTrait for &'a Box<MyType> { }

fn main() { }
