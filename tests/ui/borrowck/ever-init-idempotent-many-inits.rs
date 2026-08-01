//! Many reassignments of the same local must not change borrowck's
//! Boolean ever-init checks or illegal-reassignment diagnostics.
//!
//! Guards the EverInitializedPlaces optimization that omits later InitIndex
//! facts for a move path that is already ever-initialized in the lattice.

//@ run-rustfix

fn many_mut_ok(mut acc: u64) -> u64 {
    // Conditional backedges so MIR keeps cycles; each iteration reassigns `acc`.
    for i in 0..32u64 {
        if i % 2 == 0 {
            acc = acc.wrapping_add(i);
        } else {
            acc = acc.wrapping_mul(3).wrapping_add(1);
        }
    }
    acc
}

fn main() {
    let _ = many_mut_ok(1);

    let v: i32;
    //~^ HELP consider making this binding mutable
    //~| SUGGESTION mut
    v = 1;
    //~^ NOTE first assignment
    let _ = v;
    v = 2;
    //~^ ERROR cannot assign twice to immutable variable
    //~| NOTE cannot assign twice to immutable
    let _ = v;
}
