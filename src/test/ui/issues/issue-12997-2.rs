// compile-flags: --test

//! Test that makes sure wrongly-typed bench functions are rejected

#[bench]
fn bar(x: isize) { }
//~^ ERROR mismatched types
