// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that destructor on a struct runs successfully after the struct
// is boxed and converted to an object.

static mut value: uint = 0;

struct Cat {
    name : uint,
}

trait Dummy {
    fn get(&self) -> uint;
}

impl Dummy for Cat {
    fn get(&self) -> uint { self.name }
}

impl Drop for Cat {
    fn drop(&mut self) {
        unsafe { value = self.name; }
    }
}

pub fn main() {
    {
        let x = box Cat {name: 22};
        let nyan: Box<Dummy> = x as Box<Dummy>;
    }
    unsafe {
        assert_eq!(value, 22);
    }
}
