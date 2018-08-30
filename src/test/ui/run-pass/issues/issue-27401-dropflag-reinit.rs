// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty issue #37201

// Check that when a `let`-binding occurs in a loop, its associated
// drop-flag is reinitialized (to indicate "needs-drop" at the end of
// the owning variable's scope).

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
        if i == 1 { // Note that
            break;  //  both this break
        }           //   and also
        drop(a);    //    this move of `a`
        // are necessary to expose the bug
    }
    assert_eq!(cnt, 2);
}
