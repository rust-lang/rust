//! Companion tests to `divergent-guard-bindings.rs`: even if a match guard can't fail after
//! mutating the scrutinee, it can't use deref patterns to bind sub-places of the mutated place by
//! value if the arm body is reachable.

#![feature(deref_patterns)]

fn main() {
    // In the match arms below, `x` can't be bound after the assignment in the guard.

    let mut v = &mut vec![vec![false]];
    let empty = &mut vec![];
    match v {
        &mut [[x]] if { *v = vec![]; true } || return => {}
        //~^ ERROR cannot assign to `*v` because it is borrowed
        &mut [[x]] if { v = empty; true } || return => {}
        //~^ ERROR cannot assign `v` in match guard
        _ => {}
    }

    let mut b = &mut Box::new(Box::new(false));
    let t = &mut Box::new(Box::new(true));
    match b {
        &mut deref!(deref!(x)) if { ***b = true; true } || return => {}
        //~^ ERROR cannot assign to `***b` because it is borrowed
        &mut deref!(deref!(x)) if { b = t; true } || return => {}
        //~^ ERROR cannot assign `b` in match guard
        _ => {}
    }
}
