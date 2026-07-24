// Test that `rustc_specialization_marker` is only allowed on marker traits.

#![feature(rustc_attrs)]

#[unsafe(rustc_specialization_marker)]
trait SpecMarker {
    fn f();
    //~^ ERROR marker traits
}

#[unsafe(rustc_specialization_marker)]
trait SpecMarker2 {
    type X;
    //~^ ERROR marker traits
}

fn main() {}
