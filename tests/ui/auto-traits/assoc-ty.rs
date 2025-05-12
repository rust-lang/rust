//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Tests that projection doesn't explode if we accidentally
// put an associated type on an auto trait.

auto trait Trait {
    //~^ ERROR auto traits are experimental and possibly buggy
    type Output;
    //~^ ERROR auto traits cannot have associated items
}

fn main() {
    let _: <() as Trait>::Output = ();
    //[current]~^ ERROR mismatched types
    //[next]~^^ ERROR type mismatch resolving `<() as Trait>::Output normalizes-to _`
    //[next]~| ERROR type mismatch resolving `<() as Trait>::Output normalizes-to _`
}
