//@ compile-flags:-Zverbose-internals

#![allow(warnings)]

use std::fmt::Debug;

fn no_region<'a, T>(x: Box<T>) -> impl Debug + 'a
where
    T: Debug,
{
    x
    //~^ ERROR the parameter type `T` may not live long enough [E0309]
}

fn correct_region<'a, T>(x: Box<T>) -> impl Debug + 'a
where
    T: 'a + Debug,
{
    x
}

fn wrong_region<'a, 'b, T>(x: Box<T>) -> impl Debug + 'a
where
    T: 'b + Debug,
{
    x
    //~^ ERROR the parameter type `T` may not live long enough [E0309]
}

fn outlives_region<'a, 'b, T>(x: Box<T>) -> impl Debug + 'a
where
    T: 'b + Debug,
    'b: 'a,
{
    x
}

fn main() {}
