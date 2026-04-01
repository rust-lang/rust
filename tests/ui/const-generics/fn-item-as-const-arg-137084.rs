// Regression test for https://github.com/rust-lang/rust/issues/137084
// Previously caused ICE when using function item as const generic argument

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

fn a<const b: i32>() {}
fn d(e: &String) {
    a::<d>
    //~^ ERROR mismatched types
    //~| ERROR the constant `d` is not of type `i32`
}

fn main() {}
