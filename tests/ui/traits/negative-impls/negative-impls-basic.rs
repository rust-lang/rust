// A simple test that we are able to create negative impls, when the
// feature gate is given.
//
// build-pass

#![feature(negative_impls)]


struct TestType;

trait TestTrait {
    fn dummy(&self) {}
}

impl !TestTrait for TestType {}

fn main() {}
