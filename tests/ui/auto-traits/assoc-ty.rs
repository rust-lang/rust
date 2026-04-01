//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Tests that projection doesn't explode if we accidentally
// put an associated type on an auto trait.

auto trait Trait {
    //~^ ERROR auto traits are experimental and possibly buggy
    //~| HELP add `#![feature(auto_traits)]` to the crate attributes to enable
    type Output;
    //~^ ERROR auto traits cannot have associated items
    //~| HELP remove the associated items
}

fn main() {
    let _: <() as Trait>::Output = ();
    //[current]~^ ERROR mismatched types
    //[current]~| HELP consider constraining the associated type `<() as Trait>::Output` to `()` or calling a method that returns `<() as Trait>::Output`
    //[next]~^^^ ERROR type mismatch resolving `<() as Trait>::Output normalizes-to _`
    //[next]~| ERROR type mismatch resolving `<() as Trait>::Output normalizes-to _`
}
