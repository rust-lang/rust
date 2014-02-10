// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Thingy {
    x: int,
    y: int
}

impl ToStr for Thingy {
    fn to_str(&self) -> ~str {
        format!("\\{ x: {}, y: {} \\}", self.x, self.y)
    }
}

struct PolymorphicThingy<T> {
    x: T
}

impl<T:ToStr> ToStr for PolymorphicThingy<T> {
    fn to_str(&self) -> ~str {
        self.x.to_str()
    }
}

pub fn main() {
    println!("{}", Thingy { x: 1, y: 2 }.to_str());
    println!("{}", PolymorphicThingy { x: Thingy { x: 1, y: 2 } }.to_str());
}
