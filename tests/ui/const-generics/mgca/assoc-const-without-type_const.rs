#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

pub trait Tr {
    const SIZE: usize;
}

fn mk_array<T: Tr>(_x: T) -> [(); T::SIZE] {
    //~^ ERROR: use of `const` in the type system not defined as `type const`
    [(); T::SIZE]
    //~^ ERROR: use of `const` in the type system not defined as `type const`
}

fn main() {}
