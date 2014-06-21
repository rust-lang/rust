// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

struct Thingy {
    x: int,
    y: int
}

impl fmt::Show for Thingy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{ x: {}, y: {} }}", self.x, self.y)
    }
}

struct PolymorphicThingy<T> {
    x: T
}

impl<T:fmt::Show> fmt::Show for PolymorphicThingy<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.x)
    }
}

pub fn main() {
    println!("{}", Thingy { x: 1, y: 2 }.to_string());
    println!("{}", PolymorphicThingy { x: Thingy { x: 1, y: 2 } }.to_string());
}
