// This test causes ERROR: mismatched types [E0308]
// and makes rustc to print array from const arguments
#![feature(min_generic_const_args, adt_const_params)]
#![allow(incomplete_features)]

struct TakesArr<const N: [u8; 1]>;

fn foo<const N: u8>() {
    let _: TakesArr<{ [N] }> = TakesArr::<{ [1] }>;
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
