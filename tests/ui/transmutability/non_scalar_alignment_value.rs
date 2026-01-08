#![feature(min_generic_const_args)]
//~^ WARN the feature `min_generic_const_args` is incomplete

#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};
    struct Dst {}
    fn is_maybe_transmutable()
    where
        Dst: TransmuteFrom<
            (),
            {
                Assume {
                    alignment: Assume {},
                    //~^ ERROR struct expression with missing field initialiser for `alignment`
                    //~| ERROR struct expression with missing field initialiser for `lifetimes`
                    //~| ERROR struct expression with missing field initialiser for `safety`
                    //~| ERROR struct expression with missing field initialiser for `validity`
                    lifetimes: const { true },
                    safety: const { true },
                    validity: const { true },
                }
            },
        >,
    {
    }
}

fn main() {}
