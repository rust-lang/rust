// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct _X([u8]);

impl std::ops::Deref for _X {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

pub fn _g(x: &_X) -> &[u8] {
    x
}

fn main() {
}
