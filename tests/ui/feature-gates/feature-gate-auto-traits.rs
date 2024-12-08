// Test that default and negative trait implementations are gated by
// `auto_traits` feature gate

struct DummyStruct;

auto trait AutoDummyTrait {}
//~^ ERROR auto traits are experimental and possibly buggy

impl !AutoDummyTrait for DummyStruct {}
//~^ ERROR negative trait bounds are not yet fully implemented; use marker types for now

fn main() {}
