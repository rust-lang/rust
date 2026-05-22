//! Test that `-Z deduplicate-diagnostics` flag properly deduplicates diagnostic messages.

//@ revisions: duplicate deduplicate
//@[deduplicate] compile-flags: -Z deduplicate-diagnostics=yes

#[derive(Unresolved)] //~ ERROR cannot find derive macro `Unresolved` in this scope
                      //[duplicate]~| ERROR cannot find derive macro `Unresolved` in this scope
struct S;

#[deny("literal")] //~ ERROR malformed `deny` attribute input [E0539]
fn main() {}
