pub type Archived<T> = <m::Alias as n::Trait>::Archived;
//~^ ERROR failed to resolve: use of undeclared crate or module `m`
//~| ERROR failed to resolve: use of undeclared crate or module `n`

fn main() {}
