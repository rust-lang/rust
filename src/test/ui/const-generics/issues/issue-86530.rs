#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait X {
    const Y: usize;
}

fn z<T>(t: T)
where
    T: X,
    [(); T::Y]: ,
{
}

fn unit_literals() {
    z(" ");
    //~^ ERROR: the trait bound `&str: X` is not satisfied
}

fn main() {}
