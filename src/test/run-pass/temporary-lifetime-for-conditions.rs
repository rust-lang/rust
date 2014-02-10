// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// This test verifies that temporaries created for `while`'s
// and `if` conditions are correctly cleaned up.

struct Temporary;

static mut DROPPED: int = 0;

impl Drop for Temporary {
    fn drop(&mut self) {
        unsafe { DROPPED += 1; }
    }
}

impl Temporary {
    fn do(&self) -> bool {true}
}

fn borrow() -> ~Temporary { ~Temporary }


pub fn main() {
    let mut i = 0;

    // This loop's condition
    // should call `Temporary`'s
    // `drop` 6 times.
    while borrow().do() {
        i += 1;
        if i > 5 {
            break;
        }
    }

    // This if condition should
    // call it 1 time
    if borrow().do() {
        unsafe { assert_eq!(DROPPED, 7) }
    }
}
