// aux-build:suggest-constructor-cycle-error.rs
//~^ cycle detected when getting the resolver for lowering [E0391]

// Regression test for https://github.com/rust-lang/rust/issues/119625

extern crate suggest_constructor_cycle_error as a;

const CONST_NAME: a::Uuid = a::Uuid(());

fn main() {}
