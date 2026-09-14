//! Regression test for #157141.
//!
//! A non-`move`, non-`use` closure that only `.use`s an upvar should capture it
//! by immutable borrow, not by `use` (which would clone the value into the
//! closure at construction time). The `.use` expression in the body is then the
//! only thing that clones, once per evaluation.

//@ run-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::sync::atomic::{AtomicUsize, Ordering};

static CLONES: AtomicUsize = AtomicUsize::new(0);

struct Thing;

impl Clone for Thing {
    fn clone(&self) -> Self {
        CLONES.fetch_add(1, Ordering::Relaxed);
        Thing
    }
}

impl std::clone::UseCloned for Thing {}

fn clones_during<R>(f: impl FnOnce() -> R) -> usize {
    let before = CLONES.load(Ordering::Relaxed);
    f();
    CLONES.load(Ordering::Relaxed) - before
}

fn main() {
    // Plain `||` closure: `x` is borrowed, so building the closure clones nothing.
    // Each call clones exactly once via the `.use` in the body.
    let x = Thing;
    let n = clones_during(|| {
        let closure = || {
            let _y = x.use;
        };
        closure();
        closure();
    });
    assert_eq!(n, 2, "plain `||` closure should clone once per call, not also at capture");
    drop(x);

    // `move ||` closure: `x` is moved in (no capture clone), `.use` clones per call.
    let x = Thing;
    let n = clones_during(|| {
        let closure = move || {
            let _y = x.use;
        };
        closure();
        closure();
    });
    assert_eq!(n, 2, "`move ||` closure should clone once per call");

    // `use ||` closure: the capture clause clones `x` into the closure once, at
    // construction. Calling it afterwards moves the captured value out.
    let x = Thing;
    let n = clones_during(|| {
        let closure = use || {
            let _y = &x;
        };
        closure();
    });
    assert_eq!(n, 1, "`use ||` closure clones once at construction");
}
