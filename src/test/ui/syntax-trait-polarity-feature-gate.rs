use std::marker::Send;

struct TestType;

trait TestTrait {}

impl !Send for TestType {}
//~^ ERROR negative trait bounds

fn main() {}
