//! Clone a coroutine from a dependency with `coroutine_clone`, without the feature.
//@ aux-build: clone-cross-crate-aux.rs
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver
//@ run-pass

#![feature(coroutine_trait)]

extern crate clone_cross_crate_aux as aux;

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut a = aux::make();
    let mut b = a.clone();
    assert_eq!(Pin::new(&mut a).resume(()), CoroutineState::Yielded(1));
    assert_eq!(Pin::new(&mut b).resume(()), CoroutineState::Yielded(1));
}
