// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::{Deref, DerefMut};

/// Takes ownership of `T` and tracks whether it was accessed mutably
/// (via `DerefMut`). You can access this via the `maybe_mutated` fn.
#[derive(Clone, Debug)]
pub struct AccessTracker<T> {
    value: T,
    mutated: bool,
}

impl<T> AccessTracker<T> {
    pub fn new(value: T) -> Self {
        AccessTracker { value, mutated: false }
    }

    /// True if the owned value was accessed mutably (so far).
    pub fn maybe_mutated(this: &Self) -> bool {
        this.mutated
    }

    pub fn into_inner(this: Self) -> (T, bool) {
        (this.value, this.mutated)
    }
}

impl<T> Deref for AccessTracker<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for AccessTracker<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.mutated = true;
        &mut self.value
    }
}

