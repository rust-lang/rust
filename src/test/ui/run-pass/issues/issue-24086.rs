// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Registry<'a> {
    listener: &'a mut (),
}

pub struct Listener<'a> {
    pub announce: Option<Box<FnMut(&mut Registry) + 'a>>,
    pub remove: Option<Box<FnMut(&mut Registry) + 'a>>,
}

impl<'a> Drop for Registry<'a> {
    fn drop(&mut self) {}
}

fn main() {
    let mut registry_listener = Listener {
        announce: None,
        remove: None,
    };
}
