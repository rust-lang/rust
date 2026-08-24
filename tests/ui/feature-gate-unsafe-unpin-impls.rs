#![feature(unsafe_unpin, negative_impls)]
use std::marker::UnsafeUnpin;

struct MyType;

unsafe impl UnsafeUnpin for MyType {}
impl !UnsafeUnpin for MyType {}

fn main() {}
