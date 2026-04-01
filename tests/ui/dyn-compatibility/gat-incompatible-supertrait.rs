// Test that the dyn-compatibility diagnostics for GATs refer first to the
// user-named trait, not the GAT-containing supertrait.
//
// NOTE: this test is currently broken, and first reports:
// "the trait `Super` is not dyn compatible"
//
//@ edition:2018

trait Super {
    type Assoc<'a>;
}

trait Child: Super {}

fn take_dyn(_: &dyn Child) {}
//~^ ERROR the trait `Super` is not dyn compatible

fn main() {}
