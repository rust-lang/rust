// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The stack-local drop flag for binding in match-arm `(_, a) => ...`
// (that occurred in a loop) were failing to (re-)initialize the drop
// flag after it had been set to "moved" during an earlier iteration
// of the loop.
//
// This is a regression test to ensure we don't let that behavior
// creep back in.
//
// See also issue-27401-let-init.rs

use std::cell::Cell;

struct A<'a>(&'a Cell<i32>);

impl<'a> Drop for A<'a> {
    fn drop(&mut self) {
        let old_val = self.0.get();
        self.0.set(old_val + 1);
    }
}

fn main() {
    let cnt = Cell::new(0);
    for i in 0..2 {
        match (A(&cnt), A(&cnt))  {
            (_, aaah) => {
                if i == 1 {
                    break
                }
                drop(aaah);
            }
        }
    }
    assert_eq!(cnt.get(), 4);
}
