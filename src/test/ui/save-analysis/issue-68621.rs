// compile-flags: -Zsave-analysis

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait Trait {}

trait Service {
    type Future: Trait;
}

struct Struct;

impl Service for Struct {
    type Future = impl Trait; //~ ERROR: could not find defining uses
}

fn main() {}
