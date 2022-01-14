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

trait T: Default {
    fn f()
    where
        Self: Default;
}

trait U: Default {
    fn f()
    where
        Self: Clone;
}

trait ZZ: Default {
    fn g();
    fn h();
    fn f()
    where
        Self: Default + Clone;
}

trait BadTrait: Default + Clone {
    fn f()
    where
        Self: Default + Clone;
    fn g()
    where
        Self: Default;
    fn h()
    where
        Self: Copy;
}

#[derive(Default, Clone)]
struct Life {}

impl T for Life {
    // this should not warn
    fn f() {}
}

impl U for Life {
    // this should not warn
    fn f() {}
}

fn main() {}
