pub type Archived<T> = <m::Alias as n::Trait>::Archived;
//~^ ERROR cannot find item `n` in this scope
//~| ERROR cannot find item `m` in this scope

fn main() {}
