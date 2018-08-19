// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Tests that automatic coercions from &mut T to *mut T
// allow borrows of T to expire immediately - essentially, that
// they work identically to 'foo as *mut T'
#![feature(nll)]

struct SelfReference {
    self_reference: *mut SelfReference,
}

impl SelfReference {
    fn set_self_ref(&mut self) {
        self.self_reference = self;
    }
}

fn main() {}
