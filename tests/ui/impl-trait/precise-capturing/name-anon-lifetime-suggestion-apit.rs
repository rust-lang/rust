//@ edition: 2021
//@ run-rustfix

#![allow(dead_code)]

fn no_generics(x: &(), y: &(), _z: impl Sized) -> impl Sized {
    (x, y)
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn with_type_param<U>(x: &(), y: &(), _z: impl Sized, _u: U) -> impl Sized {
    (x, y)
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn with_lifetime_param<'a>(x: &'a (), y: &(), z: &(), _w: impl Sized) -> impl Sized {
    (x, y, z)
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn two_apits(x: &(), y: &(), _a: impl Sized, _b: impl Copy) -> impl Sized {
    (x, y)
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn one_anon(x: &(), _z: impl Sized) -> impl Sized {
    x
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

fn main() {}
