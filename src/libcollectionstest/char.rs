// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use collections::vec::Vec;

#[test]
fn char_to_lowercase() {
    assert_iter_eq('A'.to_lowercase(), &['a']);
    assert_iter_eq('É'.to_lowercase(), &['é']);
    assert_iter_eq('ǅ'.to_lowercase(), &['ǆ']);
}

#[test]
fn char_to_uppercase() {
    assert_iter_eq('a'.to_uppercase(), &['A']);
    assert_iter_eq('é'.to_uppercase(), &['É']);
    assert_iter_eq('ǅ'.to_uppercase(), &['Ǆ']);
    assert_iter_eq('ß'.to_uppercase(), &['S', 'S']);
    assert_iter_eq('ﬁ'.to_uppercase(), &['F', 'I']);
    assert_iter_eq('ᾀ'.to_uppercase(), &['Ἀ', 'Ι']);
}

fn assert_iter_eq<I: Iterator<Item=char>>(iter: I, expected: &[char]) {
    assert_eq!(iter.collect::<Vec<_>>(), expected);
}
