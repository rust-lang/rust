// run-pass

#![feature(matches_macro)]

use std::macros::matches;

fn main() {
    let foo = 'f';
    assert!(matches!(foo, 'A'..='Z' | 'a'..='z'));

    let foo = '_';
    assert!(!matches!(foo, 'A'..='Z' | 'a'..='z'));
}
