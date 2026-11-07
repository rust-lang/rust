// Test that specializing on a `rustc_allow_lifetime_dependent_specialization` trait is
// allowed.

//@ check-pass

#![feature(min_specialization)]
#![feature(rustc_attrs)]

#[unsafe(rustc_allow_lifetime_dependent_specialization)]
trait SpecMarker {}

trait X {
    fn f();
}

impl<T> X for T {
    default fn f() {}
}

impl<T: SpecMarker> X for T {
    fn f() {}
}

fn main() {}
