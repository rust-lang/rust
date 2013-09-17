// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct thing { x: int, }

impl Drop for thing {
    fn drop(&mut self) {}
}

fn thing() -> thing {
    thing {
        x: 0
    }
}

impl thing {
    pub fn f(self) {}
}

pub fn main() {
    let z = thing();
    (z).f();
}
