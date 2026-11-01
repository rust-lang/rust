// Test that specializing on a trait is not allowed in general.

#![feature(min_specialization)]

trait SpecMarker {}

trait X {
    fn f();
}

impl<T> X for T {
    default fn f() {}
}

impl<T: SpecMarker> X for T {
    //~^ ERROR cannot specialize on trait `SpecMarker`
    fn f() {}
}

fn main() {}
