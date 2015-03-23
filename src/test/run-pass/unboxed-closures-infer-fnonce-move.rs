// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(unsafe_destructor)]

// Test that we are able to infer a suitable kind for this `move`
// closure that is just called (`FnOnce`).

use std::mem;

struct DropMe<'a>(&'a mut i32);

#[unsafe_destructor]
impl<'a> Drop for DropMe<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn main() {
    let mut counter = 0;

    {
        let drop_me = DropMe(&mut counter);
        let tick = move || mem::drop(drop_me);
        tick();
    }

    assert_eq!(counter, 1);
}
