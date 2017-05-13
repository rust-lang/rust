// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code for checking whether the output of the compiler matches what is
//! expected.

pub fn diff_lines(actual: &str, expected: &str) -> Vec<String> {
    // mega simplistic diff algorithm that just prints the things added/removed
    zip_all(actual.lines(), expected.lines())
        .enumerate()
        .filter_map(|(i, (a, e))| {
            match (a, e) {
                (Some(a), Some(e)) => {
                    if lines_match(e, a) {
                        None
                    } else {
                        Some(format!("{:3} - |{}|\n    + |{}|\n", i, e, a))
                    }
                }
                (Some(a), None) => Some(format!("{:3} -\n    + |{}|\n", i, a)),
                (None, Some(e)) => Some(format!("{:3} - |{}|\n    +\n", i, e)),
                (None, None) => panic!("Cannot get here"),
            }
        })
        .collect()
}

fn lines_match(expected: &str, mut actual: &str) -> bool {
    for (i, part) in expected.split("[..]").enumerate() {
        match actual.find(part) {
            Some(j) => {
                if i == 0 && j != 0 {
                    return false;
                }
                actual = &actual[j + part.len()..];
            }
            None => return false,
        }
    }
    actual.is_empty() || expected.ends_with("[..]")
}

struct ZipAll<I1: Iterator, I2: Iterator> {
    first: I1,
    second: I2,
}

impl<T, I1: Iterator<Item = T>, I2: Iterator<Item = T>> Iterator for ZipAll<I1, I2> {
    type Item = (Option<T>, Option<T>);
    fn next(&mut self) -> Option<(Option<T>, Option<T>)> {
        let first = self.first.next();
        let second = self.second.next();

        match (first, second) {
            (None, None) => None,
            (a, b) => Some((a, b)),
        }
    }
}

fn zip_all<T, I1: Iterator<Item = T>, I2: Iterator<Item = T>>(a: I1, b: I2) -> ZipAll<I1, I2> {
    ZipAll {
        first: a,
        second: b,
    }
}
