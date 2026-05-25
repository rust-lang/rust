#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

type const CONST: usize = 1_i32;
                        //~^ ERROR the literal is not of type `usize`

fn main() {
    const { CONST };
}
