// Regression test for https://github.com/rust-lang/rust/issues/160490.

fn f(...: u8) {}
//~^ ERROR unexpected `...`
//~| ERROR missing pattern for `...` argument
//~| WARN this was previously accepted by the compiler

fn main() {}
