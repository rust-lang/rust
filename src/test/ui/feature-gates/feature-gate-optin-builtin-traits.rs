// Test that default and negative trait implementations are gated by
// `optin_builtin_traits` feature gate

struct DummyStruct;

trait DummyTrait {
    fn dummy(&self) {}
}

auto trait AutoDummyTrait {}
//~^ ERROR auto traits are experimental and possibly buggy

impl !DummyTrait for DummyStruct {}
//~^ ERROR negative trait bounds are not yet fully implemented; use marker types for now

fn main() {}
