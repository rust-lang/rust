#![feature(optin_builtin_traits)]

struct TestType;

trait TestTrait {
    fn dummy(&self) { }
}

impl !TestTrait for TestType {}
//~^ ERROR invalid negative impl

fn main() {}
