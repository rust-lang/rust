//! Regression test for https://github.com/rust-lang/rust/issues/13359

//@ dont-require-annotations: NOTE

fn foo(_s: i16) { }

fn bar(_s: u32) { }

fn main() {
    foo(1*(1 as isize));
    //~^ ERROR mismatched types
    //~| NOTE expected `i16`, found `isize`

    bar(1*(1 as usize));
    //~^ ERROR mismatched types
    //~| NOTE expected `u32`, found `usize`
}
