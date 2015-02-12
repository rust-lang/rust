// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

pub struct TreeBuilder<H> { pub h: H }

impl<H> TreeBuilder<H> {
    pub fn process_token(&mut self) {
        match self {
            _ => for _y in self.by_ref() {}
        }
    }
}

impl<H> Iterator for TreeBuilder<H> {
    type Item = H;

    fn next(&mut self) -> Option<H> {
        None
    }
}
