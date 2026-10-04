// Regression test for https://github.com/rust-lang/rust/issues/157892

#![cfg_attr(a())]
//~^ ERROR malformed `cfg_attr` attribute input [E0539]
//~| ERROR malformed `cfg_attr` attribute input [E0539]

fn main() {}
