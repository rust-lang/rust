#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

type const CONST: usize = 1_i32;
//~^ ERROR the constant `1` is not of type `usize`
//~| NOTE expected `usize`, found `i32`

fn main() {
    const { CONST };
}
