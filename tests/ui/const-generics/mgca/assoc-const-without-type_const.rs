#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

pub trait Tr {
    const SIZE: usize;
}

fn mk_array<T: Tr>(_x: T) -> [(); T::SIZE] {
    //~^ ERROR type_const
    [(); T::SIZE]
    //~^ ERROR type_const
}

fn main() {}
