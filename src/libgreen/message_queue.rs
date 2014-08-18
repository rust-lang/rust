// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::arc::Arc;
use std::sync::mpsc_queue as mpsc;
use std::kinds::marker;

pub enum PopResult<T> {
    Inconsistent,
    Empty,
    Data(T),
}

pub fn queue<T: Send>() -> (Consumer<T>, Producer<T>) {
    let a = Arc::new(mpsc::Queue::new());
    (Consumer { inner: a.clone(), noshare: marker::NoSync },
     Producer { inner: a, noshare: marker::NoSync })
}

pub struct Producer<T> {
    inner: Arc<mpsc::Queue<T>>,
    noshare: marker::NoSync,
}

pub struct Consumer<T> {
    inner: Arc<mpsc::Queue<T>>,
    noshare: marker::NoSync,
}

impl<T: Send> Consumer<T> {
    pub fn pop(&self) -> PopResult<T> {
        match self.inner.pop() {
            mpsc::Inconsistent => Inconsistent,
            mpsc::Empty => Empty,
            mpsc::Data(t) => Data(t),
        }
    }

    pub fn casual_pop(&self) -> Option<T> {
        match self.inner.pop() {
            mpsc::Inconsistent => None,
            mpsc::Empty => None,
            mpsc::Data(t) => Some(t),
        }
    }
}

impl<T: Send> Producer<T> {
    pub fn push(&self, t: T) {
        self.inner.push(t);
    }
}

impl<T: Send> Clone for Producer<T> {
    fn clone(&self) -> Producer<T> {
        Producer { inner: self.inner.clone(), noshare: marker::NoSync }
    }
}
