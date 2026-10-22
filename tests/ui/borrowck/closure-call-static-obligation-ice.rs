//! Regression test for https://github.com/rust-lang/rust/issues/164477

use std::marker::PhantomData;

struct Bounded<'b: 'static>(PhantomData<&'b ()>);

fn hr<F: for<'x> Fn(&'x ()) -> Bounded<'x>>(f: F) -> F {
    f
}

fn g<'a>(x: &'a ()) {
    let f = hr(|_v| Bounded(PhantomData));
    f(x);
    //~^ ERROR borrowed data escapes outside of function
}

fn main() {}
