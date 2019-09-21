// Test that default and negative trait implementations are gated by
// `optin_builtin_traits` feature gate

struct DummyStruct;

auto trait AutoDummyTrait {}
//~^ ERROR auto traits are experimental and possibly buggy

impl !AutoDummyTrait for DummyStruct {}
//~^ ERROR negative trait bounds are not yet fully implemented; use marker types for now

macro_rules! accept_item { ($i:item) => {} }
accept_item! {
    auto trait Auto {}
    //~^ ERROR auto traits are experimental and possibly buggy
}
accept_item! {
    impl !AutoDummyTrait for DummyStruct {}
    //~^ ERROR negative trait bounds are not yet fully implemented; use marker types for now
}
fn main() {}
