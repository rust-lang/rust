#![feature(optin_builtin_traits)]

struct TestType;

trait TestTrait {
    fn dummy(&self) { }
}

impl !TestTrait for TestType {}
//~^ ERROR negative impls are only allowed for auto traits (e.g., `Send` and `Sync`)

fn main() {}
