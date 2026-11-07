#![feature(gca_min_const_items, gca_macroless_args)]
#![allow(incomplete_features)]

pub trait Tr {
    const SIZE: usize;
}

fn mk_array<T: Tr>(_x: T) -> [(); T::SIZE] {
    //~^ ERROR: use of `const` in the type system not marked as direct
    [(); T::SIZE]
    //~^ ERROR: use of `const` in the type system not marked as direct
}

fn main() {}
