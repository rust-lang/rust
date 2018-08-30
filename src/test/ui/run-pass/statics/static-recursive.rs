// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static mut S: *const u8 = unsafe { &S as *const *const u8 as *const u8 };

struct StaticDoubleLinked {
    prev: &'static StaticDoubleLinked,
    next: &'static StaticDoubleLinked,
    data: i32,
    head: bool
}

static L1: StaticDoubleLinked = StaticDoubleLinked{prev: &L3, next: &L2, data: 1, head: true};
static L2: StaticDoubleLinked = StaticDoubleLinked{prev: &L1, next: &L3, data: 2, head: false};
static L3: StaticDoubleLinked = StaticDoubleLinked{prev: &L2, next: &L1, data: 3, head: false};


pub fn main() {
    unsafe { assert_eq!(S, *(S as *const *const u8)); }

    let mut test_vec = Vec::new();
    let mut cur = &L1;
    loop {
        test_vec.push(cur.data);
        cur = cur.next;
        if cur.head { break }
    }
    assert_eq!(&test_vec, &[1,2,3]);

    let mut test_vec = Vec::new();
    let mut cur = &L1;
    loop {
        cur = cur.prev;
        test_vec.push(cur.data);
        if cur.head { break }
    }
    assert_eq!(&test_vec, &[3,2,1]);
}
