// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]

use std::cell::RefCell;

pub struct Window<Data>{
    pub data: RefCell<Data>
}

impl<Data:  Update> Window<Data> {
    pub fn update(&self, e: i32) {
        match e {
            1 => self.data.borrow_mut().update(),
            _ => {}
        }
    }
}

pub trait Update {
    fn update(&mut self);
}
