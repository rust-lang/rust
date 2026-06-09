//@compile-flags: --crate-name borrow_interior_mutable_const_ignore
//@check-pass

#![warn(clippy::borrow_interior_mutable_const)]
#![allow(clippy::declare_interior_mutable_const)]

use core::cell::Cell;
use std::cmp::{Eq, PartialEq};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counted<T> {
    count: AtomicUsize,
    val: T,
}

impl<T> Counted<T> {
    const fn new(val: T) -> Self {
        Self {
            count: AtomicUsize::new(0),
            val,
        }
    }
}

enum OptionalCell {
    Unfrozen(Counted<bool>),
    Frozen,
}

const UNFROZEN_VARIANT: OptionalCell = OptionalCell::Unfrozen(Counted::new(true));
const FROZEN_VARIANT: OptionalCell = OptionalCell::Frozen;

fn main() {
    let _ = &UNFROZEN_VARIANT;
}
