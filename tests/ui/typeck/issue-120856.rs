pub type Archived<T> = <m::Alias as n::Trait>::Archived;
//~^ ERROR cannot find item `m`
//~| NOTE use of undeclared crate or module `m`
//~| ERROR cannot find item `n`
//~| NOTE use of undeclared crate or module `n`

fn main() {}
