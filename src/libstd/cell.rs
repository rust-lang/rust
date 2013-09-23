// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A mutable, nullable memory location

#[missing_doc];

use cast::transmute_mut;
use clone::{Clone, DeepClone};
use cmp::Eq;
use option::{Option, Some, None};

/*
A dynamic, mutable location.

Similar to a mutable option type, but friendlier.
*/

#[no_freeze]
#[deriving(Clone, DeepClone, Eq)]
#[allow(missing_doc)]
pub struct Cell<T> {
    priv value: Option<T>
}

impl<T> Cell<T> {
    /// Creates a new full cell with the given value.
    pub fn new(value: T) -> Cell<T> {
        Cell { value: Some(value) }
    }

    /// Creates a new empty cell with no value inside.
    pub fn new_empty() -> Cell<T> {
        Cell { value: None }
    }

    /// Yields the value, failing if the cell is empty.
    pub fn take(&self) -> T {
        unsafe {
            let mvalue = transmute_mut(&self.value);
            mvalue.take_unwrap()
        }
    }

    /// Returns the value, failing if the cell is full.
    pub fn put_back(&self, value: T) {
        assert!(self.value.is_none());
        unsafe {
            let mvalue = transmute_mut(&self.value);
            *mvalue = Some(value);
        }
    }

    /// Returns true if the cell is empty and false if the cell is full.
    pub fn is_empty(&self) -> bool {
        self.value.is_none()
    }
}

#[test]
fn test_basic() {
    let value_cell = Cell::new(~10);
    assert!(!value_cell.is_empty());
    let value = value_cell.take();
    assert!(value == ~10);
    assert!(value_cell.is_empty());
    value_cell.put_back(value);
    assert!(!value_cell.is_empty());
}

#[test]
#[should_fail]
fn test_take_empty() {
    let value_cell: Cell<~int> = Cell::new_empty();
    value_cell.take();
}

#[test]
#[should_fail]
fn test_put_back_non_empty() {
    let value_cell = Cell::new(~10);
    value_cell.put_back(~20);
}
