// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Tests that we can call a function bounded over a supertrait from
// a default method

fn require_y<T: Y>(x: T) -> int { x.y() }

trait Y {
    fn y(self) -> int;
}


trait Z: Y {
    fn x(self) -> int {
        require_y(self)
    }
}

impl Y for int {
    fn y(self) -> int { self }
}

impl Z for int {}

fn main() {
    assert_eq!(12.x(), 12);
}
