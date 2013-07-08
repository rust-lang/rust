// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::uint::{range};

trait FooTrait {
    fn foo(&self) -> uint;
}

struct BarStruct {
    x: uint
}

impl FooTrait for BarStruct {
    fn foo(&self) -> uint {
        self.x
    }
}

pub fn main() {
    let foos: ~[ ~FooTrait ] = ~[
        ~BarStruct{ x: 0 } as ~FooTrait,
        ~BarStruct{ x: 1 } as ~FooTrait,
        ~BarStruct{ x: 2 } as ~FooTrait
    ];

    for range(0, foos.len()) |i| {
        assert_eq!(i, foos[i].foo());
    }
}
