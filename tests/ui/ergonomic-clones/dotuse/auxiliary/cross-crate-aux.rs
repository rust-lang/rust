#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;
use std::sync::atomic::{AtomicUsize, Ordering};

static CLONES: AtomicUsize = AtomicUsize::new(0);

#[derive(Copy)]
pub struct S;

impl Clone for S {
    fn clone(&self) -> S {
        CLONES.fetch_add(1, Ordering::SeqCst);
        S
    }
}

impl UseCloned for S {}

pub fn dup<T: UseCloned>(x: T) -> (T, T) {
    (x.use, x)
}

pub fn clones() -> usize {
    CLONES.load(Ordering::SeqCst)
}
