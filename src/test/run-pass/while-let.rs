// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(binary_heap_extras)]

use std::collections::BinaryHeap;

fn make_pq() -> BinaryHeap<isize> {
    BinaryHeap::from(vec![1,2,3])
}

pub fn main() {
    let mut pq = make_pq();
    let mut sum = 0;
    while let Some(x) = pq.pop() {
        sum += x;
    }
    assert_eq!(sum, 6);

    pq = make_pq();
    sum = 0;
    'a: while let Some(x) = pq.pop() {
        sum += x;
        if x == 2 {
            break 'a;
        }
    }
    assert_eq!(sum, 5);

    pq = make_pq();
    sum = 0;
    'a: while let Some(x) = pq.pop() {
        if x == 3 {
            continue 'a;
        }
        sum += x;
    }
    assert_eq!(sum, 3);

    let mut pq1 = make_pq();
    sum = 0;
    while let Some(x) = pq1.pop() {
        let mut pq2 = make_pq();
        while let Some(y) = pq2.pop() {
            sum += x * y;
        }
    }
    assert_eq!(sum, 6 + 12 + 18);
}
