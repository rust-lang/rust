use std::cell::Cell;
use std::cmp::Ordering::{self, *};
use std::ptr;

// Minimal type with an `Ord` implementation violating transitivity.
#[derive(Debug)]
pub(crate) enum Cyclic3 {
    A,
    B,
    C,
}
use Cyclic3::*;

impl PartialOrd for Cyclic3 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cyclic3 {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (A, A) | (B, B) | (C, C) => Equal,
            (A, B) | (B, C) | (C, A) => Less,
            (A, C) | (B, A) | (C, B) => Greater,
        }
    }
}

impl PartialEq for Cyclic3 {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(&other) == Equal
    }
}

impl Eq for Cyclic3 {}

// Controls the ordering of values wrapped by `Governed`.
#[derive(Debug)]
pub(crate) struct Governor {
    flipped: Cell<bool>,
}

impl Governor {
    pub(crate) fn new() -> Self {
        Governor { flipped: Cell::new(false) }
    }

    pub(crate) fn flip(&self) {
        self.flipped.set(!self.flipped.get());
    }
}

// Type with an `Ord` implementation that forms a total order at any moment
// (assuming that `T` respects total order), but can suddenly be made to invert
// that total order.
#[derive(Debug)]
pub(crate) struct Governed<'a, T>(pub T, pub &'a Governor);

impl<T: Ord> PartialOrd for Governed<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for Governed<'_, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        assert!(ptr::eq(self.1, other.1));
        let ord = self.0.cmp(&other.0);
        if self.1.flipped.get() { ord.reverse() } else { ord }
    }
}

impl<T: PartialEq> PartialEq for Governed<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        assert!(ptr::eq(self.1, other.1));
        self.0.eq(&other.0)
    }
}

impl<T: Eq> Eq for Governed<'_, T> {}
