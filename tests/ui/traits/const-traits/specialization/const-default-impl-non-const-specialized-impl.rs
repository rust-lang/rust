// Tests that specializing trait impls must be at least as const as the default impl.
//@ revisions: spec min_spec

#![feature(const_trait_impl)]
#![cfg_attr(spec, feature(specialization))]
//[spec]~^ WARN the feature `specialization` is incomplete
#![cfg_attr(min_spec, feature(min_specialization))]

#[const_trait]
trait Value {
    fn value() -> u32;
}

impl<T> const Value for T {
    default fn value() -> u32 {
        0
    }
}

struct FortyTwo;

impl Value for FortyTwo {
    //~^ ERROR conflicting implementations
    fn value() -> u32 {
        println!("You can't do that (constly)");
        42
    }
}

fn main() {}
