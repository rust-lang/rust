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
    //~^ ERROR trait `X` is not implemented for `&str`
}

fn main() {}
