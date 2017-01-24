// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hash::{self, Hash};
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone)]
pub struct RcSlice<T> {
    data: Rc<Box<[T]>>,
    offset: u32,
    len: u32,
}

impl<T> RcSlice<T> {
    pub fn new(vec: Vec<T>) -> Self {
        RcSlice {
            offset: 0,
            len: vec.len() as u32,
            data: Rc::new(vec.into_boxed_slice()),
        }
    }
}

impl<T> Deref for RcSlice<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.data[self.offset as usize .. (self.offset + self.len) as usize]
    }
}

impl<T: Hash> Hash for RcSlice<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl<T: fmt::Debug> fmt::Debug for RcSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}
