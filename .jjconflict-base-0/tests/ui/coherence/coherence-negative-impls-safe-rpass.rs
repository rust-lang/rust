//@ run-pass
#![allow(dead_code)]

#![feature(negative_impls)]

struct TestType;

impl !Send for TestType {}

fn main() {}
