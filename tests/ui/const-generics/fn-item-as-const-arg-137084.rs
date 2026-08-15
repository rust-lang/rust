// Regression test for https://github.com/rust-lang/rust/issues/137084
// Previously caused ICE when using function item as const generic argument

#![feature(min_generic_const_args, macroless_generic_const_args)]
#![allow(incomplete_features)]

fn a<const b: i32>() {}
fn d(e: &String) {
    a::<d>
    //~^ ERROR function items cannot be used as const args
}

fn main() {}
