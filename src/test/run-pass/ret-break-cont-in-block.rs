// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

use std::cmp::Eq;
use std::vec;

fn iter<T>(v: ~[T], it: &fn(&T) -> bool) -> bool {
    let mut i = 0u, l = v.len();
    while i < l {
        if !it(&v[i]) { return false; }
        i += 1u;
    }
    return true;
}

fn find_pos<T:Eq + Copy + Clone>(n: T, h: ~[T]) -> Option<uint> {
    let mut i = 0u;
    for iter(h.clone()) |e| {
        if *e == n { return Some(i); }
        i += 1u;
    }
    None
}

fn bail_deep(x: ~[~[bool]]) {
    let mut seen = false;
    for iter(x.clone()) |x| {
        for iter(x.clone()) |x| {
            assert!(!seen);
            if *x { seen = true; return; }
        }
    }
    assert!(!seen);
}

fn ret_deep() -> ~str {
    for iter(~[1, 2]) |e| {
        for iter(~[3, 4]) |x| {
            if *e + *x > 4 { return ~"hi"; }
        }
    }
    return ~"bye";
}

pub fn main() {
    let mut last = 0;
    for vec::all(~[1, 2, 3, 4, 5, 6, 7]) |e| {
        last = *e;
        if *e == 5 { break; }
        if *e % 2 == 1 { loop; }
        assert_eq!(*e % 2, 0);
    };
    assert_eq!(last, 5);

    assert_eq!(find_pos(1, ~[0, 1, 2, 3]), Some(1u));
    assert_eq!(find_pos(1, ~[0, 4, 2, 3]), None);
    assert_eq!(find_pos(~"hi", ~[~"foo", ~"bar", ~"baz", ~"hi"]), Some(3u));

    bail_deep(~[~[false, false], ~[true, true], ~[false, true]]);
    bail_deep(~[~[true]]);
    bail_deep(~[~[false, false, false]]);

    assert_eq!(ret_deep(), ~"hi");
}
