// Test that `rustc_allow_lifetime_dependent_specialization` is only allowed on marker traits.

#![feature(rustc_attrs)]

#[unsafe(rustc_allow_lifetime_dependent_specialization)]
trait SpecMarker {
    fn f();
    //~^ ERROR marker traits
}

#[unsafe(rustc_allow_lifetime_dependent_specialization)]
trait SpecMarker2 {
    type X;
    //~^ ERROR marker traits
}

fn main() {}
