import core::*;

use std;

import unicode;

#[test]
fn test_is_digit() {
    assert (unicode::icu::is_digit('0'));
    assert (!unicode::icu::is_digit('m'));
}

#[test]
fn test_is_lower() {
    assert (unicode::icu::is_lower('m'));
    assert (!unicode::icu::is_lower('M'));
}

#[test]
fn test_is_space() {
    assert (unicode::icu::is_space(' '));
    assert (!unicode::icu::is_space('m'));
}

#[test]
fn test_is_upper() {
    assert (unicode::icu::is_upper('M'));
    assert (!unicode::icu::is_upper('m'));
}

