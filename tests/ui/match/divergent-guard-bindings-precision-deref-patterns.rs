//! Companion tests to `divergent-guard-bindings-precision.rs`: it's an error to mutate a matched-on
//! place in a guard that can succeed, even if it can't fail and no sub-places of the mutated place
//! are bound by value. These are separated out because they depend on an unstable feature.

#![feature(deref_patterns)]

fn main() {
    // In the match arms below, `x` is bound by reference before evaluating the guard. Since the
    // arms' bindings can be created and it's impossible to continue matching after the guards, it
    // would technically be sound to permit these.

    let mut v = &mut vec![vec![false]];
    let empty = &mut vec![];
    match v {
        // We match on `*v.index(0).index(0)`.
        [[true]] => {}

        // The assignment would conflict with `x`'s borrow, so `x` can't be used thereafter.
        &mut [[ref x]] if { v[0][0] = true; true } || return => {}
        //~^ ERROR: cannot borrow `*v` as mutable because it is also borrowed as immutable

        // `x` could be used after the assignment; it would still point to the original `false`.
        &mut [[ref x]] if { v = empty; true } || return => { *x; }
        //~^ ERROR: cannot assign `v` in match guard

        _ => {}
    }

    let mut b = &mut Box::new(Box::new(false));
    let t = &mut Box::new(Box::new(true));
    match b {
        // We match on `***b`.
        true => {}

        // The assignment would conflict with `x`'s borrow, so `x` can't be used thereafter.
        &mut deref!(deref!(ref x)) if { ***b = true; true } || return => {}
        //~^ ERROR: cannot assign `***b` in match guard

        // `x` could be used after the assignment; it would still point to the original `false`.
        &mut deref!(deref!(ref x)) if { b = t; true } || return => { *x; }
        //~^ ERROR: cannot assign `b` in match guard

        _ => {}
    }
}
