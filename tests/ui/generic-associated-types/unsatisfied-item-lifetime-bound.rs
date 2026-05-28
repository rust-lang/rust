#![warn(unused_lifetimes, redundant_lifetimes)]

pub trait X {
    type Y<'a: 'static>; //~ WARN unnecessary lifetime parameter `'a`
}

impl X for () {
    type Y<'a> = &'a ();
    //~^ ERROR lifetime bound not satisfied
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
