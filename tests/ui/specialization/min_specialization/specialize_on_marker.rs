// Test that specializing on a `rustc_unsafe_specialization_marker` trait is
// allowed.

//@ check-pass

#![feature(min_specialization)]
#![feature(rustc_attrs)]

#[rustc_unsafe_specialization_marker]
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
