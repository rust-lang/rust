// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// A version is either an exact revision,
/// or a semantic version

extern mod std;

use std::char;

pub type Version = Option<~str>;

// Being lazy since we don't have a regexp library now
#[deriving(Eq)]
enum ParseState {
    Start,
    SawDigit,
    SawDot
}

pub fn try_parsing_version(s: &str) -> Option<~str> {
    let s = s.trim();
    debug!("Attempting to parse: {}", s);
    let mut parse_state = Start;
    for c in s.chars() {
        if char::is_digit(c) {
            parse_state = SawDigit;
        }
        else if c == '.' && parse_state == SawDigit {
            parse_state = SawDot;
        }
        else {
            return None;
        }
    }
    match parse_state {
        SawDigit => Some(s.to_owned()),
        _        => None
    }
}

/// If s is of the form foo#bar, where bar is a valid version
/// number, return the prefix before the # and the version.
/// Otherwise, return None.
pub fn split_version<'a>(s: &'a str) -> Option<(&'a str, Version)> {
    // Check for extra '#' characters separately
    if s.split('#').len() > 2 {
        return None;
    }
    split_version_general(s, '#')
}

pub fn split_version_general<'a>(s: &'a str, sep: char) -> Option<(&'a str, Version)> {
    match s.rfind(sep) {
        Some(i) => {
            let path = s.slice(0, i);
            // n.b. for now, assuming an exact revision is intended, not a SemVer
            Some((path, Some(s.slice(i + 1, s.len()).to_owned())))
        }
        None => {
            None
        }
    }
}

#[test]
fn test_parse_version() {
    assert!(try_parsing_version("1.2") == Some(~"1.2"));
    assert!(try_parsing_version("1.0.17") == Some(~"1.0.17"));
    assert!(try_parsing_version("you're_a_kitty") == None);
    assert!(try_parsing_version("42..1") == None);
    assert!(try_parsing_version("17") == Some(~"17"));
    assert!(try_parsing_version(".1.2.3") == None);
    assert!(try_parsing_version("2.3.") == None);
}

#[test]
fn test_split_version() {
    let s = "a/b/c#0.1";
    debug!("== {:?} ==", split_version(s));
    assert!(split_version(s) == Some((s.slice(0, 5), Some(~"0.1"))));
    assert!(split_version("a/b/c") == None);
    let s = "a#1.2";
    assert!(split_version(s) == Some((s.slice(0, 1), Some(~"1.2"))));
    assert!(split_version("a#a#3.4") == None);
}
