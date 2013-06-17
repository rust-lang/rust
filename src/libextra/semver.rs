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

#[allow(missing_doc)];

use core::prelude::*;

use core::char;
use core::cmp;
use core::io::{ReaderUtil};
use core::io;
use core::option::{Option, Some, None};
use core::to_str::ToStr;
use core::uint;

#[deriving(Eq)]
pub enum Identifier {
    Numeric(uint),
    AlphaNumeric(~str)
}

impl cmp::Ord for Identifier {
    #[inline(always)]
    fn lt(&self, other: &Identifier) -> bool {
        match (self, other) {
            (&Numeric(a), &Numeric(b)) => a < b,
            (&Numeric(_), _) => true,
            (&AlphaNumeric(ref a), &AlphaNumeric(ref b)) => *a < *b,
            (&AlphaNumeric(_), _) => false
        }
    }
    #[inline(always)]
    fn le(&self, other: &Identifier) -> bool {
        ! (other < self)
    }
    #[inline(always)]
    fn gt(&self, other: &Identifier) -> bool {
        other < self
    }
    #[inline(always)]
    fn ge(&self, other: &Identifier) -> bool {
        ! (self < other)
    }
}

impl ToStr for Identifier {
    #[inline(always)]
    fn to_str(&self) -> ~str {
        match self {
            &Numeric(n) => n.to_str(),
            &AlphaNumeric(ref s) => s.to_str()
        }
    }
}


#[deriving(Eq)]
pub struct Version {
    major: uint,
    minor: uint,
    patch: uint,
    pre: ~[Identifier],
    build: ~[Identifier],
}

impl ToStr for Version {
    #[inline(always)]
    fn to_str(&self) -> ~str {
        let s = fmt!("%u.%u.%u", self.major, self.minor, self.patch);
        let s = if self.pre.is_empty() {
            s
        } else {
            s + "-" + self.pre.map(|i| i.to_str()).connect(".")
        };
        if self.build.is_empty() {
            s
        } else {
            s + "+" + self.build.map(|i| i.to_str()).connect(".")
        }
    }
}

impl cmp::Ord for Version {
    #[inline(always)]
    fn lt(&self, other: &Version) -> bool {

        self.major < other.major ||

            (self.major == other.major &&
             self.minor < other.minor) ||

            (self.major == other.major &&
             self.minor == other.minor &&
             self.patch < other.patch) ||

            (self.major == other.major &&
             self.minor == other.minor &&
             self.patch == other.patch &&
             // NB: semver spec says 0.0.0-pre < 0.0.0
             // but the version of ord defined for vec
             // says that [] < [pre], so we alter it
             // here.
             (match (self.pre.len(), other.pre.len()) {
                 (0, 0) => false,
                 (0, _) => false,
                 (_, 0) => true,
                 (_, _) => self.pre < other.pre
             })) ||

            (self.major == other.major &&
             self.minor == other.minor &&
             self.patch == other.patch &&
             self.pre == other.pre &&
             self.build < other.build)
    }

    #[inline(always)]
    fn le(&self, other: &Version) -> bool {
        ! (other < self)
    }
    #[inline(always)]
    fn gt(&self, other: &Version) -> bool {
        other < self
    }
    #[inline(always)]
    fn ge(&self, other: &Version) -> bool {
        ! (self < other)
    }
}

condition! {
    bad_parse: () -> ();
}

fn take_nonempty_prefix(rdr: @io::Reader,
                        ch: char,
                        pred: &fn(char) -> bool) -> (~str, char) {
    let mut buf = ~"";
    let mut ch = ch;
    while pred(ch) {
        buf.push_char(ch);
        ch = rdr.read_char();
    }
    if buf.is_empty() {
        bad_parse::cond.raise(())
    }
    debug!("extracted nonempty prefix: %s", buf);
    (buf, ch)
}

fn take_num(rdr: @io::Reader, ch: char) -> (uint, char) {
    let (s, ch) = take_nonempty_prefix(rdr, ch, char::is_digit);
    match uint::from_str(s) {
        None => { bad_parse::cond.raise(()); (0, ch) },
        Some(i) => (i, ch)
    }
}

fn take_ident(rdr: @io::Reader, ch: char) -> (Identifier, char) {
    let (s,ch) = take_nonempty_prefix(rdr, ch, char::is_alphanumeric);
    if s.iter().all(char::is_digit) {
        match uint::from_str(s) {
            None => { bad_parse::cond.raise(()); (Numeric(0), ch) },
            Some(i) => (Numeric(i), ch)
        }
    } else {
        (AlphaNumeric(s), ch)
    }
}

fn expect(ch: char, c: char) {
    if ch != c {
        bad_parse::cond.raise(())
    }
}

fn parse_reader(rdr: @io::Reader) -> Version {
    let (major, ch) = take_num(rdr, rdr.read_char());
    expect(ch, '.');
    let (minor, ch) = take_num(rdr, rdr.read_char());
    expect(ch, '.');
    let (patch, ch) = take_num(rdr, rdr.read_char());

    let mut pre = ~[];
    let mut build = ~[];

    let mut ch = ch;
    if ch == '-' {
        loop {
            let (id, c) = take_ident(rdr, rdr.read_char());
            pre.push(id);
            ch = c;
            if ch != '.' { break; }
        }
    }

    if ch == '+' {
        loop {
            let (id, c) = take_ident(rdr, rdr.read_char());
            build.push(id);
            ch = c;
            if ch != '.' { break; }
        }
    }

    Version {
        major: major,
        minor: minor,
        patch: patch,
        pre: pre,
        build: build,
    }
}


pub fn parse(s: &str) -> Option<Version> {
    if !s.is_ascii() {
        return None;
    }
    let s = s.trim();
    let mut bad = false;
    do bad_parse::cond.trap(|_| { debug!("bad"); bad = true }).in {
        do io::with_str_reader(s) |rdr| {
            let v = parse_reader(rdr);
            if bad || v.to_str() != s.to_owned() {
                None
            } else {
                Some(v)
            }
        }
    }
}

#[test]
fn test_parse() {
    assert_eq!(parse(""), None);
    assert_eq!(parse("  "), None);
    assert_eq!(parse("1"), None);
    assert_eq!(parse("1.2"), None);
    assert_eq!(parse("1.2"), None);
    assert_eq!(parse("1"), None);
    assert_eq!(parse("1.2"), None);
    assert_eq!(parse("1.2.3-"), None);
    assert_eq!(parse("a.b.c"), None);
    assert_eq!(parse("1.2.3 abc"), None);

    assert!(parse("1.2.3") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[],
        build: ~[],
    }));
    assert!(parse("  1.2.3  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[],
        build: ~[],
    }));
    assert!(parse("1.2.3-alpha1") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[AlphaNumeric(~"alpha1")],
        build: ~[]
    }));
    assert!(parse("  1.2.3-alpha1  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[AlphaNumeric(~"alpha1")],
        build: ~[]
    }));
    assert!(parse("1.2.3+build5") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[],
        build: ~[AlphaNumeric(~"build5")]
    }));
    assert!(parse("  1.2.3+build5  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[],
        build: ~[AlphaNumeric(~"build5")]
    }));
    assert!(parse("1.2.3-alpha1+build5") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[AlphaNumeric(~"alpha1")],
        build: ~[AlphaNumeric(~"build5")]
    }));
    assert!(parse("  1.2.3-alpha1+build5  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[AlphaNumeric(~"alpha1")],
        build: ~[AlphaNumeric(~"build5")]
    }));
    assert!(parse("1.2.3-1.alpha1.9+build5.7.3aedf  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: ~[Numeric(1),AlphaNumeric(~"alpha1"),Numeric(9)],
        build: ~[AlphaNumeric(~"build5"),
                 Numeric(7),
                 AlphaNumeric(~"3aedf")]
    }));

}

#[test]
fn test_eq() {
    assert_eq!(parse("1.2.3"), parse("1.2.3"));
    assert_eq!(parse("1.2.3-alpha1"), parse("1.2.3-alpha1"));
}

#[test]
fn test_ne() {
    assert!(parse("0.0.0")       != parse("0.0.1"));
    assert!(parse("0.0.0")       != parse("0.1.0"));
    assert!(parse("0.0.0")       != parse("1.0.0"));
    assert!(parse("1.2.3-alpha") != parse("1.2.3-beta"));
}

#[test]
fn test_lt() {
    assert!(parse("0.0.0")        < parse("1.2.3-alpha2"));
    assert!(parse("1.0.0")        < parse("1.2.3-alpha2"));
    assert!(parse("1.2.0")        < parse("1.2.3-alpha2"));
    assert!(parse("1.2.3-alpha1") < parse("1.2.3"));
    assert!(parse("1.2.3-alpha1") < parse("1.2.3-alpha2"));
    assert!(!(parse("1.2.3-alpha2") < parse("1.2.3-alpha2")));
}

#[test]
fn test_le() {
    assert!(parse("0.0.0")        <= parse("1.2.3-alpha2"));
    assert!(parse("1.0.0")        <= parse("1.2.3-alpha2"));
    assert!(parse("1.2.0")        <= parse("1.2.3-alpha2"));
    assert!(parse("1.2.3-alpha1") <= parse("1.2.3-alpha2"));
    assert!(parse("1.2.3-alpha2") <= parse("1.2.3-alpha2"));
}

#[test]
fn test_gt() {
    assert!(parse("1.2.3-alpha2") > parse("0.0.0"));
    assert!(parse("1.2.3-alpha2") > parse("1.0.0"));
    assert!(parse("1.2.3-alpha2") > parse("1.2.0"));
    assert!(parse("1.2.3-alpha2") > parse("1.2.3-alpha1"));
    assert!(parse("1.2.3")        > parse("1.2.3-alpha2"));
    assert!(!(parse("1.2.3-alpha2") > parse("1.2.3-alpha2")));
}

#[test]
fn test_ge() {
    assert!(parse("1.2.3-alpha2") >= parse("0.0.0"));
    assert!(parse("1.2.3-alpha2") >= parse("1.0.0"));
    assert!(parse("1.2.3-alpha2") >= parse("1.2.0"));
    assert!(parse("1.2.3-alpha2") >= parse("1.2.3-alpha1"));
    assert!(parse("1.2.3-alpha2") >= parse("1.2.3-alpha2"));
}

#[test]
fn test_spec_order() {

    let vs = ["1.0.0-alpha",
              "1.0.0-alpha.1",
              "1.0.0-beta.2",
              "1.0.0-beta.11",
              "1.0.0-rc.1",
              "1.0.0-rc.1+build.1",
              "1.0.0",
              "1.0.0+0.3.7",
              "1.3.7+build",
              "1.3.7+build.2.b8f12d7",
              "1.3.7+build.11.e0f985a"];
    let mut i = 1;
    while i < vs.len() {
        let a = parse(vs[i-1]).get();
        let b = parse(vs[i]).get();
        assert!(a < b);
        i += 1;
    }
}
