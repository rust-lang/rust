// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure we apply the usual coercions to return expressions

pub struct Foo<'a> {
    a: &'a mut int,
    b: int,
    take_a: bool
}

impl<'a> Foo<'a> {
    fn take(&'a mut self) -> &'a mut int {
        if self.take_a {
            self.a
        } else {
            &mut self.b
        }
    }
}

fn main() {}
