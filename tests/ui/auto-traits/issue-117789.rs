#![deny(suspicious_auto_trait_impls)]

auto trait Trait<P> {} //~ ERROR auto traits cannot have generic parameters
//~^ ERROR auto traits are experimental and possibly buggy
impl<P> Trait<P> for () {}

fn main() {}
