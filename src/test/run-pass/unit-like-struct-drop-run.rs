// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure the destructor is run for unit-like structs.

use std::boxed::BoxAny;
use std::thread::Thread;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        panic!("This panic should happen.");
    }
}

pub fn main() {
    let x = Thread::scoped(move|| {
        let _b = Foo;
    }).join();

    let s = x.err().unwrap().downcast::<&'static str>().ok().unwrap();
    assert_eq!(s.as_slice(), "This panic should happen.");
}
