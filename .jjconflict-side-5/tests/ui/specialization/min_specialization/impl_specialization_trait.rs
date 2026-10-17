// Check that specialization traits can't be implemented without a feature.

// gate-test-min_specialization

//@ aux-build:specialization-trait.rs

extern crate specialization_trait;

struct A {}

impl specialization_trait::SpecTrait for A {
    //~^ ERROR implementing `rustc_specialization_trait` traits is unstable
    fn method(&self) {}
}

fn main() {}
