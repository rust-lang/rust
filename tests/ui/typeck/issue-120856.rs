pub type Archived<T> = <m::Alias as n::Trait>::Archived;
//~^ ERROR failed to resolve: use of unresolved module or unlinked crate `m`
//~| ERROR failed to resolve: use of unresolved module or unlinked crate `n`

fn main() {}
