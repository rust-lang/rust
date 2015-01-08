// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #20582. This test caused an ICE related to
// inconsistent region erasure in trans.

struct Foo<'a> {
    buf: &'a[u8]
}

impl<'a> Iterator for Foo<'a> {
    type Item = &'a[u8];

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        Some(self.buf)
    }
}

fn main() {
}
