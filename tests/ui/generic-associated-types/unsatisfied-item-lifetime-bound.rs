#![warn(unused_lifetimes)]

pub trait X {
    type Y<'a: 'static>;
    //~^ WARNING unnecessary lifetime parameter
}

impl X for () {
    type Y<'a> = &'a ();
}

struct B<'a, T: for<'r> X<Y<'r> = &'r ()>> {
    f: <T as X>::Y<'a>,
    //~^ ERROR lifetime bound not satisfied
}

struct C<'a, T: X> {
    f: <T as X>::Y<'a>,
    //~^ ERROR lifetime bound not satisfied
}

struct D<'a> {
    f: <() as X>::Y<'a>,
    //~^ ERROR lifetime bound not satisfied
}

fn main() {}
