// This is a regression test for https://github.com/rust-lang/rust/issues/148453
// We want the `cannot find attribute` error to appear before `is not an unsafe attribute`
//@ edition: 2024

#[unsafe(does_not_exist)]
//~^ ERROR cannot find attribute
//~| ERROR is not an unsafe attribute
fn aa() {}

fn main() {}
