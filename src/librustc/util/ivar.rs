// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::Cell;

pub struct Ivar<T:Copy> {
    data: Cell<Option<T>>
}

impl<T:Copy> Ivar<T> {
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

    pub fn fulfilled(&self) -> bool {
        self.data.get().is_some()
    }

    pub fn unwrap(&self) -> T {
        self.get().unwrap()
    }
}
