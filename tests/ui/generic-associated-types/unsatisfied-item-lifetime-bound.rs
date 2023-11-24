#![warn(unused_lifetimes)]

pub trait X {
    type Y<'a: 'static>;
    //~^ WARNING unnecessary lifetime parameter
}

impl X for () {
    type Y<'a> = &'a ();
    //~^ ERROR lifetime bound not satisfied
}

// FIXME(aliemjay): this field type should be an error.
struct B<'a, T: for<'r> X<Y<'r> = &'r ()>> {
    f: <T as X>::Y<'a>,
}

struct C<'a, T: X> {
    f: <T as X>::Y<'a>,
    //~^ ERROR lifetime bound not satisfied
}

// FIXME(aliemjay): this field type should be an error.
struct D<'a> {
    f: <() as X>::Y<'a>,
}

fn main() {}
