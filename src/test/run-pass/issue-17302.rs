// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static mut DROPPED: [bool, ..2] = [false, false];

struct A(uint);
struct Foo { _a: A, _b: int }

impl Drop for A {
    fn drop(&mut self) {
        let A(i) = *self;
        unsafe { DROPPED[i] = true; }
    }
}

fn main() {
    {
        Foo {
            _a: A(0),
            ..Foo { _a: A(1), _b: 2 }
        };
    }
    unsafe {
        assert!(DROPPED[0]);
        assert!(DROPPED[1]);
    }
}
