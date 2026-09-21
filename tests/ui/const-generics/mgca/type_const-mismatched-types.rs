#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

const FREE: u32 = core::direct_const_arg!(5_usize);
//~^ ERROR the constant `5` is not of type `u32`

const FREE2: isize = core::direct_const_arg!(FREE);
//~^ ERROR the constant `5` is not of type `u32`
//~| ERROR the constant `5` is not of type `isize`

trait Tr {
    #[rustc_always_gca]
    const N: usize;
}

impl Tr for () {
    const N: usize = core::direct_const_arg!(false);
    //~^ ERROR the constant `false` is not of type `usize`
}

fn main() {}
