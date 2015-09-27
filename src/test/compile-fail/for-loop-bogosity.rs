// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct MyStruct {
    x: isize,
    y: isize,
}

impl MyStruct {
    fn next(&mut self) -> Option<isize> {
        Some(self.x)
    }
}

pub fn main() {
    let mut bogus = MyStruct {
        x: 1,
        y: 2,
    };
    for x in bogus { //~ ERROR `core::iter::Iterator` is not implemented for the type `MyStruct`
        drop(x);
    }
}
