// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;

pub fn lev_distance(me: &str, t: &str) -> usize {
    if me.is_empty() { return t.chars().count(); }
    if t.is_empty() { return me.chars().count(); }

    let mut dcol: Vec<_> = (0..t.len() + 1).collect();
    let mut t_last = 0;

    for (i, sc) in me.chars().enumerate() {

        let mut current = i;
        dcol[0] = current + 1;

        for (j, tc) in t.chars().enumerate() {

            let next = dcol[j + 1];

            if sc == tc {
                dcol[j + 1] = current;
            } else {
                dcol[j + 1] = cmp::min(current, next);
                dcol[j + 1] = cmp::min(dcol[j + 1], dcol[j]) + 1;
            }

            current = next;
            t_last = j;
        }
    }

    dcol[t_last + 1]
}

#[test]
fn test_lev_distance() {
    use std::char::{ from_u32, MAX };
    // Test bytelength agnosticity
    for c in (0..MAX as u32)
             .filter_map(|i| from_u32(i))
             .map(|i| i.to_string()) {
        assert_eq!(lev_distance(&c[..], &c[..]), 0);
    }

    let a = "\nMäry häd ä little lämb\n\nLittle lämb\n";
    let b = "\nMary häd ä little lämb\n\nLittle lämb\n";
    let c = "Mary häd ä little lämb\n\nLittle lämb\n";
    assert_eq!(lev_distance(a, b), 1);
    assert_eq!(lev_distance(b, a), 1);
    assert_eq!(lev_distance(a, c), 2);
    assert_eq!(lev_distance(c, a), 2);
    assert_eq!(lev_distance(b, c), 1);
    assert_eq!(lev_distance(c, b), 1);
}
