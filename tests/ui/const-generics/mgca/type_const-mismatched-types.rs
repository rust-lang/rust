#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

use std::gca;

const FREE: u32 = gca!(5_usize);
//~^ ERROR the constant `5` is not of type `u32`

const FREE2: isize = gca!(FREE);
//~^ ERROR the constant `5` is not of type `u32`
//~| ERROR the constant `5` is not of type `isize`

trait Tr {
    #[rustc_always_gca]
    const N: usize;
}

impl Tr for () {
    const N: usize = gca!(false);
    //~^ ERROR the constant `false` is not of type `usize`
}

fn main() {}
