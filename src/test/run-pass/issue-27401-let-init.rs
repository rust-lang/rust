// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The stack-local drop flag associated with binding `let a = ...;`
// (that occurred in a loop) were failing to (re-)initialize the drop
// flag after it had been set to "moved" during an earlier iteration
// of the loop.
//
// This is a regression test to ensure we don't let that behavior
// creep back in.
//
// See also issue-27401-match-bind.rs

struct A<'a>(&'a mut i32);

impl<'a> Drop for A<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn main() {
    let mut cnt = 0;
    for i in 0..2 {
        let a = A(&mut cnt);
        if i == 1 {
            break
        }
        drop(a);
    }
    assert_eq!(cnt, 2);
}
