pub type Archived<T> = <m::Alias as n::Trait>::Archived;
//~^ ERROR: cannot find `m` in this scope
//~| ERROR: cannot find `n` in this scope
//~| NOTE: use of unresolved module or unlinked crate `m`
//~| NOTE: use of unresolved module or unlinked crate `n`

fn main() {}
