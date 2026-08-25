//@ run-rustfix

#![allow(warnings)]

fn no_restriction<T>(x: &()) -> &() {
    with_restriction::<T>(x) //~ ERROR the parameter type `T` may not live long enough
}

fn with_restriction<'b, T: 'b>(x: &'b ()) -> &'b () {
    x
}

fn main() {}
