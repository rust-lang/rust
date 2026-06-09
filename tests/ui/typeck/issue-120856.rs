pub type Archived<T> = <m::Alias as n::Trait>::Archived;
//~^ ERROR: cannot find module or crate `m` in this scope
//~| ERROR: cannot find module or crate `n` in this scope
//~| NOTE: use of unresolved module or unlinked crate `m`
//~| NOTE: use of unresolved module or unlinked crate `n`

fn main() {}
