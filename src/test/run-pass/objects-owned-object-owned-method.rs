// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test invoked `&self` methods on owned objects where the values
// closed over contain managed values. This implies that the boxes
// will have headers that must be skipped over.


trait FooTrait {
    fn foo(self: Box<Self>) -> uint;
}

struct BarStruct {
    x: uint
}

impl FooTrait for BarStruct {
    fn foo(self: Box<BarStruct>) -> uint {
        self.x
    }
}

pub fn main() {
    let foo = box BarStruct{ x: 22 } as Box<FooTrait>;
    assert_eq!(22, foo.foo());
}
