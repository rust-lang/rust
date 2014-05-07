// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct DroppableStruct;

static mut DROPPED: bool = false;

impl Drop for DroppableStruct {
    fn drop(&mut self) {
        unsafe { DROPPED = true; }
    }
}

trait MyTrait { }
impl MyTrait for Box<DroppableStruct> {}

struct Whatever { w: Box<MyTrait> }
impl  Whatever {
    fn new(w: Box<MyTrait>) -> Whatever {
        Whatever { w: w }
    }
}

fn main() {
    {
        let f = box DroppableStruct;
        let _a = Whatever::new(box f as Box<MyTrait>);
    }
    assert!(unsafe { DROPPED });
}
