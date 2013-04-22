// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A mutable, nullable memory location

use cast::transmute_mut;
use prelude::*;

/*
A dynamic, mutable location.

Similar to a mutable option type, but friendlier.
*/

pub struct Cell<T> {
    priv value: Option<T>
}

impl<T:cmp::Eq> cmp::Eq for Cell<T> {
    fn eq(&self, other: &Cell<T>) -> bool {
        (self.value) == (other.value)
    }
    fn ne(&self, other: &Cell<T>) -> bool { !self.eq(other) }
}

/// Creates a new full cell with the given value.
pub fn Cell<T>(value: T) -> Cell<T> {
    Cell { value: Some(value) }
}

pub fn empty_cell<T>() -> Cell<T> {
    Cell { value: None }
}

pub impl<T> Cell<T> {
    /// Yields the value, failing if the cell is empty.
    fn take(&self) -> T {
        let mut self = unsafe { transmute_mut(self) };
        if self.is_empty() {
            fail!(~"attempt to take an empty cell");
        }

        let mut value = None;
        value <-> self.value;
        value.unwrap()
    }

    /// Returns the value, failing if the cell is full.
    fn put_back(&self, value: T) {
        let mut self = unsafe { transmute_mut(self) };
        if !self.is_empty() {
            fail!(~"attempt to put a value back into a full cell");
        }
        self.value = Some(value);
    }

    /// Returns true if the cell is empty and false if the cell is full.
    fn is_empty(&self) -> bool {
        self.value.is_none()
    }

    // Calls a closure with a reference to the value.
    fn with_ref<R>(&self, op: &fn(v: &T) -> R) -> R {
        let v = self.take();
        let r = op(&v);
        self.put_back(v);
        r
    }

    // Calls a closure with a mutable reference to the value.
    fn with_mut_ref<R>(&self, op: &fn(v: &mut T) -> R) -> R {
        let mut v = self.take();
        let r = op(&mut v);
        self.put_back(v);
        r
    }
}

#[test]
fn test_basic() {
    let value_cell = Cell(~10);
    assert!(!value_cell.is_empty());
    let value = value_cell.take();
    assert!(value == ~10);
    assert!(value_cell.is_empty());
    value_cell.put_back(value);
    assert!(!value_cell.is_empty());
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_take_empty() {
    let value_cell = empty_cell::<~int>();
    value_cell.take();
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_put_back_non_empty() {
    let value_cell = Cell(~10);
    value_cell.put_back(~20);
}

#[test]
fn test_with_ref() {
    let good = 6;
    let c = Cell(~[1, 2, 3, 4, 5, 6]);
    let l = do c.with_ref() |v| { v.len() };
    assert!(l == good);
}

#[test]
fn test_with_mut_ref() {
    let good = ~[1, 2, 3];
    let v = ~[1, 2];
    let c = Cell(v);
    do c.with_mut_ref() |v| { v.push(3); }
    let v = c.take();
    assert!(v == good);
}
