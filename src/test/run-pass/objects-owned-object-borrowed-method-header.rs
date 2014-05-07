// ignore-pretty

// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]


// Test invoked `&self` methods on owned objects where the values
// closed over contain managed values. This implies that the boxes
// will have headers that must be skipped over.

trait FooTrait {
    fn foo(&self) -> uint;
}

struct BarStruct {
    x: @uint
}

impl FooTrait for BarStruct {
    fn foo(&self) -> uint {
        *self.x
    }
}

pub fn main() {
    let foos: Vec<Box<FooTrait:>> = vec!(
        box BarStruct{ x: @0 } as Box<FooTrait:>,
        box BarStruct{ x: @1 } as Box<FooTrait:>,
        box BarStruct{ x: @2 } as Box<FooTrait:>
    );

    for i in range(0u, foos.len()) {
        assert_eq!(i, foos.get(i).foo());
    }
}
