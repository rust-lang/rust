//! Companion tests to `divergent-guard-bindings-precision.rs`: it's fine to mutate a matched-on
//! place in a guard as long as it can't fail after mutation and no sub-places of a matched-on place
//! are bound by value. These are separated out because they involve an unstable feature.
//@ check-pass

#![feature(deref_patterns)]

fn main() {
    // In the match arms below, `x` is bound by reference before evaluating the guard.

    let mut v = &mut vec![vec![false]];
    let empty = &mut vec![];
    match v {
        // We match on `*v.index(0).index(0)`.
        [[true]] => {}

        // `x` can't be used after the assignment because that invalidates its borrow.
        &mut [[ref x]] if { v[0][0] = true; true } || return => {}

        // Reassigning `b` doesn't affect `x`, which still points at the original `false`.
        &mut [[ref x]] if { v = empty; true } || return => { *x; }

        _ => {}
    }

    let mut b = &mut Box::new(Box::new(false));
    let t = &mut Box::new(Box::new(true));
    match b {
        // We match on `***b`.
        true => {}

        // `x` can't be used after the assignment because that invalidates its borrow.
        &mut deref!(deref!(ref x)) if { ***b = true; true } || return => {}

        // Reassigning `b` doesn't affect `x`, which still points at the original `false`.
        &mut deref!(deref!(ref x)) if { b = t; true } || return => { *x; }

        _ => {}
    }
}
