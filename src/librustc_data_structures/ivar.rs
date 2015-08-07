// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::cell::Cell;

/// A write-once variable. When constructed, it is empty, and
/// can only be set once.
///
/// Ivars ensure that data that can only be initialised once. A full
/// implementation is used for concurrency and blocks on a read of an
/// unfulfilled value. This implementation is more minimal and panics
/// if you attempt to read the value before it has been set. It is also
/// not `Sync`, but may be extended in the future to be usable as a true
/// concurrency type.
///
/// The `T: Copy` bound is not strictly needed, but it is required by
/// Cell (so removing it would require using UnsafeCell), and it
/// suffices for the current purposes.
#[derive(PartialEq)]
pub struct Ivar<T: Copy> {
    data: Cell<Option<T>>
}

impl<T: Copy> Ivar<T> {
    pub fn new() -> Ivar<T> {
        Ivar {
            data: Cell::new(None)
        }
    }

    pub fn get(&self) -> Option<T> {
        self.data.get()
    }

    pub fn fulfill(&self, value: T) {
        assert!(self.data.get().is_none(),
                "Value already set!");
        self.data.set(Some(value));
    }

    pub fn is_fulfilled(&self) -> bool {
        self.data.get().is_some()
    }

    pub fn unwrap(&self) -> T {
        self.get().unwrap()
    }
}

impl<T: Copy+fmt::Debug> fmt::Debug for Ivar<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.get() {
            Some(val) => write!(f, "Ivar({:?})", val),
            None => f.write_str("Ivar(<unfulfilled>)")
        }
    }
}

impl<T: Copy> Clone for Ivar<T> {
    fn clone(&self) -> Ivar<T> {
        match self.get() {
            Some(val) => Ivar { data: Cell::new(Some(val)) },
            None => Ivar::new()
        }
    }
}
