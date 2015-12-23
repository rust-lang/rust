// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::Name;
use std::cmp;
use parse::token::InternedString;

/// To find the Levenshtein distance between two strings
pub fn lev_distance(a: &str, b: &str) -> usize {
    // cases which don't require further computation
    if a.is_empty() {
        return b.chars().count();
    } else if b.is_empty() {
        return a.chars().count();
    }

    let mut dcol: Vec<_> = (0..b.len() + 1).collect();
    let mut t_last = 0;

    for (i, sc) in a.chars().enumerate() {
        let mut current = i;
        dcol[0] = current + 1;

        for (j, tc) in b.chars().enumerate() {
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
    } dcol[t_last + 1]
}

/// To find the best match for a given string from an iterator of names
/// As a loose rule to avoid the obviously incorrect suggestions, it takes
/// an optional limit for the maximum allowable edit distance, which defaults
/// to one-third of the given word
pub fn find_best_match_for_name<'a, T>(iter_names: T,
                                       lookup: &str,
                                       dist: Option<usize>) -> Option<InternedString>
    where T: Iterator<Item = &'a Name> {
    let max_dist = dist.map_or_else(|| cmp::max(lookup.len(), 3) / 3, |d| d);
    iter_names
    .filter_map(|name| {
        let dist = lev_distance(lookup, &name.as_str());
        match dist <= max_dist {    // filter the unwanted cases
            true => Some((name.as_str(), dist)),
            false => None,
        }
    })
    .min_by_key(|&(_, val)| val)    // extract the tuple containing the minimum edit distance
    .map(|(s, _)| s)                // and return only the string
}

#[test]
fn test_lev_distance() {
    use std::char::{from_u32, MAX};
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
