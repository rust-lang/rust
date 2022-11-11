// #103762: cases that pre-1.64 rejected, but that we now allow after the fix.

// check-pass

#![allow(bare_trait_objects)]

trait Trait {}

// Case 1: Inside fn() type.
// Pre-1.64 and 1.64 rejected this, but we can allow it.
fn a1(_: fn(&(dyn Trait + '_)) -> &str) { loop {} }
fn a2(_: fn(&(Trait + '_)) -> &str) { loop {} }
fn a_closure() {
    let _ = |_: fn(&(dyn Trait + '_)) -> &str| loop {};
    let _ = |_: fn(&(Trait + '_)) -> &str| loop {};
}

fn main() {}
