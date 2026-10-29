//! Test that `-Z deduplicate-diagnostics` flag properly deduplicates diagnostic messages.

//@ revisions: duplicate deduplicate
//@[deduplicate] compile-flags: -Z deduplicate-diagnostics=yes

#[derive(Unresolved)] //~ ERROR cannot find derive macro `Unresolved` in this scope
                      //[duplicate]~| ERROR cannot find derive macro `Unresolved` in this scope
struct S;

#[unsafe(allow(unsafe(dead_code)))]
//~^ ERROR: allow` is not an unsafe attribute
//[duplicate]~| ERROR: `allow` is not an unsafe attribute
//~| ERROR: expected identifier, found keyword `unsafe`
//[duplicate]~| ERROR: expected identifier, found keyword `unsafe`
//~| ERROR: malformed lint attribute input
//[duplicate]~| ERROR: malformed lint attribute input
fn main() {}
