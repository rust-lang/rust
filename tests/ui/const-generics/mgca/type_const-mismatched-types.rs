#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

type const FREE: u32 = 5_usize;
//~^ ERROR the constant `5` is not of type `u32`

type const FREE2: isize = FREE;
//~^ ERROR the constant `5` is not of type `isize`

trait Tr {
    type const N: usize;
}

impl Tr for () {
    type const N: usize = false;
    //~^ ERROR the constant `false` is not of type `usize`
}

fn main() {}
