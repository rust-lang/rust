// Regression test of #86483.

#![feature(generic_associated_types)]
#![allow(incomplete_features)]

pub trait IceIce<T> //~ ERROR: the parameter type `T` may not live long enough
where
    for<'a> T: 'a,
{
    type Ice<'v>: IntoIterator<Item = &'v T>;
    //~^ ERROR: the parameter type `T` may not live long enough
    //~| ERROR: the parameter type `T` may not live long enough
}

fn main() {}
