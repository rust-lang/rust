//! Test that `-Z deduplicate-diagnostics` flag properly deduplicates diagnostic messages.

//@ revisions: duplicate deduplicate
//@[deduplicate] compile-flags: -Z deduplicate-diagnostics=yes

#[derive(Unresolved)] //~ ERROR cannot find derive macro `Unresolved` in this scope
                      //[duplicate]~| ERROR cannot find derive macro `Unresolved` in this scope
struct S;

#[deny("literal")] //~ ERROR expected a lint, found literal
                   //[duplicate]~| ERROR expected a lint, found literal
                   //[duplicate]~| ERROR expected a lint, found literal
fn main() {}
