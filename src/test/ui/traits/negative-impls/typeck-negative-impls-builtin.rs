// run-pass

#![feature(negative_impls)]
#![allow(dead_code)]

struct TestType;

trait TestTrait {
    fn dummy(&self) { }
}

impl !TestTrait for TestType {}

fn main() {}
