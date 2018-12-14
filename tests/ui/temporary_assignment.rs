// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::temporary_assignment)]

use std::ops::{Deref, DerefMut};

struct Struct {
    field: i32,
}

struct Wrapper<'a> {
    inner: &'a mut Struct,
}

impl<'a> Deref for Wrapper<'a> {
    type Target = Struct;
    fn deref(&self) -> &Struct {
        self.inner
    }
}

impl<'a> DerefMut for Wrapper<'a> {
    fn deref_mut(&mut self) -> &mut Struct {
        self.inner
    }
}

fn main() {
    let mut s = Struct { field: 0 };
    let mut t = (0, 0);

    Struct { field: 0 }.field = 1;
    (0, 0).0 = 1;

    // no error
    s.field = 1;
    t.0 = 1;
    Wrapper { inner: &mut s }.field = 1;
}
