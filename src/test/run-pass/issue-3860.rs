// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

struct Foo { x: int }

impl Foo {
    pub fn stuff<'a>(&'a mut self) -> &'a mut Foo {
        return self;
    }
}

pub fn main() {
    let x = @mut Foo { x: 3 };
    // Neither of the next two lines should cause an error
    let _ = x.stuff();
    x.stuff();
}
