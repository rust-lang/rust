#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

#[type_const]
const FREE: u32 = 5_usize;
//~^ ERROR mismatched types

#[type_const]
const FREE2: isize = FREE;
//~^ ERROR the constant `5` is not of type `isize`

trait Tr {
    #[type_const]
    const N: usize;
}

impl Tr for () {
    #[type_const]
    const N: usize = false;
    //~^ ERROR mismatched types
}

fn main() {}
