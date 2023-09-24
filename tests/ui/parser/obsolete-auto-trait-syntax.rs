// revisions: success failure
//[success] check-pass

trait Trait1 {}

#[cfg(failure)]
impl Trait1 for .. {} //[failure]~ ERROR `impl Trait for .. {}` is an obsolete syntax for auto traits

#[cfg(failure)]
auto trait Trait2 {} //[failure]~ ERROR `auto trait Trait {}` is an obsolete syntax for auto traits

fn main() {}
