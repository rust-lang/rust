// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![allow(unknown_features)]
#![feature(box_syntax)]

fn pairwise_sub(mut t: Box<DoubleEndedIterator<Item=isize>>) -> isize {
    let mut result = 0;
    loop {
        let front = t.next();
        let back = t.next_back();
        match (front, back) {
            (Some(f), Some(b)) => { result += b - f; }
            _ => { return result; }
        }
    }
}

fn main() {
    let v = vec![1, 2, 3, 4, 5, 6];
    // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
    let r = pairwise_sub(Box::new(v.into_iter()));
    assert_eq!(r, 9);
}
