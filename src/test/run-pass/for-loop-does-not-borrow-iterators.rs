// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// The `for` loop use to keep a mutable borrow when executing its body,
// making it impossible to re-use the iterator as follows.
// https://github.com/rust-lang/rust/issues/8372
//
// This was fixed in https://github.com/rust-lang/rust/pull/15809

pub fn main() {
    let mut for_loop_values = Vec::new();
    let mut explicit_next_call_values = Vec::new();

    let mut iter = range(1i, 10);
    for i in iter {
        for_loop_values.push(i);
        explicit_next_call_values.push(iter.next());
    }

    assert_eq!(for_loop_values, vec![1, 3, 5, 7, 9]);
    assert_eq!(explicit_next_call_values, vec![Some(2), Some(4), Some(6), Some(8), None]);
}
