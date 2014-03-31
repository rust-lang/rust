// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use mpsc = std::sync::mpsc_queue;
use std::sync::arc::UnsafeArc;

pub enum PopResult<T> {
    Inconsistent,
    Empty,
    Data(T),
}

pub fn queue<T: Send>() -> (Consumer<T>, Producer<T>) {
    let (a, b) = UnsafeArc::new2(mpsc::Queue::new());
    (Consumer { inner: a }, Producer { inner: b })
}

pub struct Producer<T> {
    inner: UnsafeArc<mpsc::Queue<T>>,
}

pub struct Consumer<T> {
    inner: UnsafeArc<mpsc::Queue<T>>,
}

impl<T: Send> Consumer<T> {
    pub fn pop(&mut self) -> PopResult<T> {
        match unsafe { (*self.inner.get()).pop() } {
            mpsc::Inconsistent => Inconsistent,
            mpsc::Empty => Empty,
            mpsc::Data(t) => Data(t),
        }
    }

    pub fn casual_pop(&mut self) -> Option<T> {
        match unsafe { (*self.inner.get()).pop() } {
            mpsc::Inconsistent => None,
            mpsc::Empty => None,
            mpsc::Data(t) => Some(t),
        }
    }
}

impl<T: Send> Producer<T> {
    pub fn push(&mut self, t: T) {
        unsafe { (*self.inner.get()).push(t); }
    }
}

impl<T: Send> Clone for Producer<T> {
    fn clone(&self) -> Producer<T> {
        Producer { inner: self.inner.clone() }
    }
}
