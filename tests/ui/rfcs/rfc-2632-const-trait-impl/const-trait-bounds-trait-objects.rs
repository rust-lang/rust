#![feature(const_trait_impl)]
// edition: 2021

#[const_trait]
trait Trait {}

fn main() {
    let _: &dyn const Trait; //~ ERROR const trait bounds are not allowed in trait object types
}
