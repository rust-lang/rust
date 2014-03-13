// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct DerefArray<'a, T> {
    inner: &'a [T]
}

impl<'a, T> Deref<&'a [T]> for DerefArray<'a, T> {
    fn deref<'b>(&'b self) -> &'b &'a [T] {
        &self.inner
    }
}

pub fn main() {
    let a = &[1, 2, 3];
    assert_eq!(DerefArray {inner: a}[1], 2);
}
