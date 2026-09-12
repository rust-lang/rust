//! Regression test for <https://github.com/rust-lang/rust/issues/57936>.
//!
//! `X` is only implemented for `fn(&'a ())` for some specific `'a`, so neither the
//! indirect call through a generic parameter nor the direct call may use it at the
//! higher-ranked type `for<'a> fn(&'a ())`. Both are rejected now; the indirect one
//! used to be accepted.

trait X {
    type G;
    fn make_g() -> Self::G;
}

impl<'a> X for fn(&'a ()) {
    type G = &'a ();

    fn make_g() -> Self::G {
        &()
    }
}

fn indirect<T: X>() {
    let x = T::make_g();
}

fn call_indirect() {
    indirect::<fn(&())>();
    //~^ ERROR implementation of `X` is not general enough
}

fn direct() {
    let x = <fn(&())>::make_g();
    //~^ ERROR no associated function or constant named `make_g` found for fn pointer `for<'a> fn(&'a ())` in the current scope
}

fn main() {}
