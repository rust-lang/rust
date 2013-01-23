// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Semver parsing and logic

use io;
use io::{ReaderUtil};
use option::{Option, Some, None};
use uint;
use str;
use to_str::ToStr;
use char;
use cmp;

pub struct Version {
    major: uint,
    minor: uint,
    patch: uint,
    tag: Option<~str>,
}

impl Version: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str {
        let suffix = match copy self.tag {
            Some(tag) => ~"-" + tag,
            None => ~""
        };

        fmt!("%u.%u.%u%s", self.major, self.minor, self.patch, suffix)
    }
}

impl Version: cmp::Ord {
    #[inline(always)]
    pure fn lt(&self, other: &Version) -> bool {
        self.major < other.major ||
        self.minor < other.minor ||
        self.patch < other.patch ||
        (match self.tag {
            Some(stag) => match other.tag {
                Some(otag) => stag < otag,
                None => true
            },
            None => false
        })
    }
    #[inline(always)]
    pure fn le(&self, other: &Version) -> bool {
        self.major <= other.major ||
        self.minor <= other.minor ||
        self.patch <= other.patch ||
        (match self.tag {
            Some(stag) => match other.tag {
                Some(otag) => stag <= otag,
                None => true
            },
            None => false
        })
    }
    #[inline(always)]
    pure fn gt(&self, other: &Version) -> bool {
        self.major > other.major ||
        self.minor > other.minor ||
        self.patch > other.patch ||
        (match self.tag {
            Some(stag) => match other.tag {
                Some(otag) => stag > otag,
                None => false
            },
            None => true
        })
    }
    #[inline(always)]
    pure fn ge(&self, other: &Version) -> bool {
        self.major >= other.major ||
        self.minor >= other.minor ||
        self.patch >= other.patch ||
        (match self.tag {
            Some(stag) => match other.tag {
                Some(otag) => stag >= otag,
                None => false
            },
            None => true
        })
    }
}

fn read_whitespace(rdr: io::Reader, ch: char) -> char {
    let mut nch = ch;

    while char::is_whitespace(nch) {
        nch = rdr.read_char();
    }

    nch
}

fn parse_reader(rdr: io::Reader) -> Option<(Version, char)> {
    fn read_digits(rdr: io::Reader, ch: char) -> Option<(uint, char)> {
        let mut buf = ~"";
        let mut nch = ch;

        while nch != -1 as char {
            match nch {
              '0' .. '9' => buf += str::from_char(nch),
              _ => break
            }

            nch = rdr.read_char();
        }

        do uint::from_str(buf).chain_ref |&i| {
            Some((i, nch))
        }
    }

    fn read_tag(rdr: io::Reader) -> Option<(~str, char)> {
        let mut ch = rdr.read_char();
        let mut buf = ~"";

        while ch != -1 as char {
            match ch {
                '0' .. '9' | 'A' .. 'Z' | 'a' .. 'z' | '-' => {
                    buf += str::from_char(ch);
                }
                _ => break
            }

            ch = rdr.read_char();
        }

        if buf == ~"" { return None; }
        else { Some((buf, ch)) }
    }

    let ch = read_whitespace(rdr, rdr.read_char());
    let (major, ch) = match read_digits(rdr, ch) {
        None => return None,
        Some(item) => item
    };

    if ch != '.' { return None; }

    let (minor, ch) = match read_digits(rdr, rdr.read_char()) {
        None => return None,
        Some(item) => item
    };

    if ch != '.' { return None; }

    let (patch, ch) = match read_digits(rdr, rdr.read_char()) {
        None => return None,
        Some(item) => item
    };
    let (tag, ch) = if ch == '-' {
        match read_tag(rdr) {
            None => return None,
            Some((tag, ch)) => (Some(tag), ch)
        }
    } else {
        (None, ch)
    };

    Some((Version { major: major, minor: minor, patch: patch, tag: tag },
          ch))
}

pub fn parse(s: &str) -> Option<Version> {
    do io::with_str_reader(s) |rdr| {
        do parse_reader(rdr).chain_ref |&item| {
            let (version, ch) = item;

            if read_whitespace(rdr, ch) != -1 as char {
                None
            } else {
                Some(version)
            }
        }
    }
}

#[test]
fn test_parse() {
    assert parse("") == None;
    assert parse("  ") == None;
    assert parse("1") == None;
    assert parse("1.2") == None;
    assert parse("1.2") == None;
    assert parse("1") == None;
    assert parse("1.2") == None;
    assert parse("1.2.3-") == None;
    assert parse("a.b.c") == None;
    assert parse("1.2.3 abc") == None;

    assert parse("1.2.3") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        tag: None,
    });
    assert parse("  1.2.3  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        tag: None,
    });
    assert parse("1.2.3-alpha1") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        tag: Some("alpha1")
    });
    assert parse("  1.2.3-alpha1  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        tag: Some("alpha1")
    });
}

#[test]
fn test_eq() {
    assert parse("1.2.3")        == parse("1.2.3");
    assert parse("1.2.3-alpha1") == parse("1.2.3-alpha1");
}

#[test]
fn test_ne() {
    assert parse("0.0.0")       != parse("0.0.1");
    assert parse("0.0.0")       != parse("0.1.0");
    assert parse("0.0.0")       != parse("1.0.0");
    assert parse("1.2.3-alpha") != parse("1.2.3-beta");
}

#[test]
fn test_lt() {
    assert parse("0.0.0")        < parse("1.2.3-alpha2");
    assert parse("1.0.0")        < parse("1.2.3-alpha2");
    assert parse("1.2.0")        < parse("1.2.3-alpha2");
    assert parse("1.2.3")        < parse("1.2.3-alpha2");
    assert parse("1.2.3-alpha1") < parse("1.2.3-alpha2");

    assert !(parse("1.2.3-alpha2") < parse("1.2.3-alpha2"));
}

#[test]
fn test_le() {
    assert parse("0.0.0")        <= parse("1.2.3-alpha2");
    assert parse("1.0.0")        <= parse("1.2.3-alpha2");
    assert parse("1.2.0")        <= parse("1.2.3-alpha2");
    assert parse("1.2.3")        <= parse("1.2.3-alpha2");
    assert parse("1.2.3-alpha1") <= parse("1.2.3-alpha2");
    assert parse("1.2.3-alpha2") <= parse("1.2.3-alpha2");
}

#[test]
fn test_gt() {
    assert parse("1.2.3-alpha2") > parse("0.0.0");
    assert parse("1.2.3-alpha2") > parse("1.0.0");
    assert parse("1.2.3-alpha2") > parse("1.2.0");
    assert parse("1.2.3-alpha2") > parse("1.2.3");
    assert parse("1.2.3-alpha2") > parse("1.2.3-alpha1");

    assert !(parse("1.2.3-alpha2") > parse("1.2.3-alpha2"));
}

#[test]
fn test_ge() {
    assert parse("1.2.3-alpha2") >= parse("0.0.0");
    assert parse("1.2.3-alpha2") >= parse("1.0.0");
    assert parse("1.2.3-alpha2") >= parse("1.2.0");
    assert parse("1.2.3-alpha2") >= parse("1.2.3");
    assert parse("1.2.3-alpha2") >= parse("1.2.3-alpha1");
    assert parse("1.2.3-alpha2") >= parse("1.2.3-alpha2");
}
