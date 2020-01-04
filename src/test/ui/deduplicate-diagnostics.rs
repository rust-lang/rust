// revisions: duplicate deduplicate
//[duplicate] compile-flags: -Z deduplicate-diagnostics=no

#[derive(Unresolved)] //~ ERROR cannot find derive macro `Unresolved` in this scope
                      //[duplicate]~| ERROR cannot find derive macro `Unresolved` in this scope
struct S;

fn main() {}
