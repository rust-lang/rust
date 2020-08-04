#![deny(clippy::trait_duplication_in_bounds)]

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

fn bad_foo<T: Clone + Default, Z: Copy>(arg0: T, arg1: Z)
where
    T: Clone,
    T: Default,
{
    unimplemented!();
}

fn good_bar<T: Clone + Default>(arg: T) {
    unimplemented!();
}

fn good_foo<T>(arg: T)
where
    T: Clone + Default,
{
    unimplemented!();
}

fn good_foobar<T: Default>(arg: T)
where
    T: Clone,
{
    unimplemented!();
}

fn main() {}
