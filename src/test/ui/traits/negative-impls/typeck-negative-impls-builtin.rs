// run-pass

#![feature(optin_builtin_traits)]
#![allow(dead_code)]

struct TestType;

trait TestTrait {
    fn dummy(&self) {}
}

impl !TestTrait for TestType {}

fn main() {}
