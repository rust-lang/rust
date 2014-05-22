// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Semantic version parsing and comparison.
//!
//! Semantic versioning (see http://semver.org/) is a set of rules for
//! assigning version numbers intended to convey meaning about what has
//! changed, and how much. A version number has five parts:
//!
//!  * Major number, updated for incompatible API changes
//!  * Minor number, updated for backwards-compatible API additions
//!  * Patch number, updated for backwards-compatible bugfixes
//!  * Pre-release information (optional), preceded by a hyphen (`-`)
//!  * Build metadata (optional), preceded by a plus sign (`+`)
//!
//! The three mandatory components are required to be decimal numbers. The
//! pre-release information and build metadata are required to be a
//! period-separated list of identifiers containing only alphanumeric
//! characters and hyphens.
//!
//! An example version number with all five components is
//! `0.8.1-rc.3.0+20130922.linux`.

#![crate_id = "semver#0.11.0-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/")]
#![deny(deprecated_owned_vector)]

use std::char;
use std::cmp;
use std::fmt;
use std::fmt::Show;
use std::option::{Option, Some, None};
use std::strbuf::StrBuf;

/// An identifier in the pre-release or build metadata. If the identifier can
/// be parsed as a decimal value, it will be represented with `Numeric`.
#[deriving(Clone, Eq)]
#[allow(missing_doc)]
pub enum Identifier {
    Numeric(uint),
    AlphaNumeric(StrBuf)
}

impl cmp::Ord for Identifier {
    #[inline]
    fn lt(&self, other: &Identifier) -> bool {
        match (self, other) {
            (&Numeric(a), &Numeric(b)) => a < b,
            (&Numeric(_), _) => true,
            (&AlphaNumeric(ref a), &AlphaNumeric(ref b)) => *a < *b,
            (&AlphaNumeric(_), _) => false
        }
    }
}

impl fmt::Show for Identifier {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Numeric(ref n) => n.fmt(f),
            AlphaNumeric(ref s) => s.fmt(f)
        }
    }
}


/// Represents a version number conforming to the semantic versioning scheme.
#[deriving(Clone)]
pub struct Version {
    /// The major version, to be incremented on incompatible changes.
    pub major: uint,
    /// The minor version, to be incremented when functionality is added in a
    /// backwards-compatible manner.
    pub minor: uint,
    /// The patch version, to be incremented when backwards-compatible bug
    /// fixes are made.
    pub patch: uint,
    /// The pre-release version identifier, if one exists.
    pub pre: Vec<Identifier>,
    /// The build metadata, ignored when determining version precedence.
    pub build: Vec<Identifier>,
}

impl fmt::Show for Version {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{}.{}.{}", self.major, self.minor, self.patch))
        if !self.pre.is_empty() {
            try!(write!(f, "-"));
            for (i, x) in self.pre.iter().enumerate() {
                if i != 0 { try!(write!(f, ".")) };
                try!(x.fmt(f));
            }
        }
        if !self.build.is_empty() {
            try!(write!(f, "+"));
            for (i, x) in self.build.iter().enumerate() {
                if i != 0 { try!(write!(f, ".")) };
                try!(x.fmt(f));
            }
        }
        Ok(())
    }
}

impl cmp::Eq for Version {
    #[inline]
    fn eq(&self, other: &Version) -> bool {
        // We should ignore build metadata here, otherwise versions v1 and v2
        // can exist such that !(v1 < v2) && !(v1 > v2) && v1 != v2, which
        // violate strict total ordering rules.
        self.major == other.major &&
            self.minor == other.minor &&
            self.patch == other.patch &&
            self.pre == other.pre
    }
}

impl cmp::Ord for Version {
    #[inline]
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
             }))
    }
}

fn take_nonempty_prefix<T:Iterator<char>>(rdr: &mut T, pred: |char| -> bool)
                        -> (StrBuf, Option<char>) {
    let mut buf = StrBuf::new();
    let mut ch = rdr.next();
    loop {
        match ch {
            None => break,
            Some(c) if !pred(c) => break,
            Some(c) => {
                buf.push_char(c);
                ch = rdr.next();
            }
        }
    }
    (buf, ch)
}

fn take_num<T: Iterator<char>>(rdr: &mut T) -> Option<(uint, Option<char>)> {
    let (s, ch) = take_nonempty_prefix(rdr, char::is_digit);
    match from_str::<uint>(s.as_slice()) {
        None => None,
        Some(i) => Some((i, ch))
    }
}

fn take_ident<T: Iterator<char>>(rdr: &mut T) -> Option<(Identifier, Option<char>)> {
    let (s,ch) = take_nonempty_prefix(rdr, char::is_alphanumeric);
    if s.as_slice().chars().all(char::is_digit) {
        match from_str::<uint>(s.as_slice()) {
            None => None,
            Some(i) => Some((Numeric(i), ch))
        }
    } else {
        Some((AlphaNumeric(s), ch))
    }
}

fn expect(ch: Option<char>, c: char) -> Option<()> {
    if ch != Some(c) {
        None
    } else {
        Some(())
    }
}

fn parse_iter<T: Iterator<char>>(rdr: &mut T) -> Option<Version> {
    let maybe_vers = take_num(rdr).and_then(|(major, ch)| {
        expect(ch, '.').and_then(|_| Some(major))
    }).and_then(|major| {
        take_num(rdr).and_then(|(minor, ch)| {
            expect(ch, '.').and_then(|_| Some((major, minor)))
        })
    }).and_then(|(major, minor)| {
        take_num(rdr).and_then(|(patch, ch)| {
           Some((major, minor, patch, ch))
        })
    });

    let (major, minor, patch, ch) = match maybe_vers {
        Some((a, b, c, d)) => (a, b, c, d),
        None => return None
    };

    let mut pre = vec!();
    let mut build = vec!();

    let mut ch = ch;
    if ch == Some('-') {
        loop {
            let (id, c) = match take_ident(rdr) {
                Some((id, c)) => (id, c),
                None => return None
            };
            pre.push(id);
            ch = c;
            if ch != Some('.') { break; }
        }
    }

    if ch == Some('+') {
        loop {
            let (id, c) = match take_ident(rdr) {
                Some((id, c)) => (id, c),
                None => return None
            };
            build.push(id);
            ch = c;
            if ch != Some('.') { break; }
        }
    }

    Some(Version {
        major: major,
        minor: minor,
        patch: patch,
        pre: pre,
        build: build,
    })
}


/// Parse a string into a semver object.
pub fn parse(s: &str) -> Option<Version> {
    if !s.is_ascii() {
        return None;
    }
    let s = s.trim();
    let v = parse_iter(&mut s.chars());
    match v {
        Some(v) => {
            if v.to_str().equiv(&s) {
                Some(v)
            } else {
                None
            }
        }
        None => None
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
        pre: vec!(),
        build: vec!(),
    }));
    assert!(parse("  1.2.3  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(),
        build: vec!(),
    }));
    assert!(parse("1.2.3-alpha1") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(AlphaNumeric("alpha1".to_strbuf())),
        build: vec!(),
    }));
    assert!(parse("  1.2.3-alpha1  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(AlphaNumeric("alpha1".to_strbuf())),
        build: vec!()
    }));
    assert!(parse("1.2.3+build5") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(),
        build: vec!(AlphaNumeric("build5".to_strbuf()))
    }));
    assert!(parse("  1.2.3+build5  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(),
        build: vec!(AlphaNumeric("build5".to_strbuf()))
    }));
    assert!(parse("1.2.3-alpha1+build5") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(AlphaNumeric("alpha1".to_strbuf())),
        build: vec!(AlphaNumeric("build5".to_strbuf()))
    }));
    assert!(parse("  1.2.3-alpha1+build5  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(AlphaNumeric("alpha1".to_strbuf())),
        build: vec!(AlphaNumeric("build5".to_strbuf()))
    }));
    assert!(parse("1.2.3-1.alpha1.9+build5.7.3aedf  ") == Some(Version {
        major: 1u,
        minor: 2u,
        patch: 3u,
        pre: vec!(Numeric(1),AlphaNumeric("alpha1".to_strbuf()),Numeric(9)),
        build: vec!(AlphaNumeric("build5".to_strbuf()),
                 Numeric(7),
                 AlphaNumeric("3aedf".to_strbuf()))
    }));

}

#[test]
fn test_eq() {
    assert_eq!(parse("1.2.3"), parse("1.2.3"));
    assert_eq!(parse("1.2.3-alpha1"), parse("1.2.3-alpha1"));
    assert_eq!(parse("1.2.3+build.42"), parse("1.2.3+build.42"));
    assert_eq!(parse("1.2.3-alpha1+42"), parse("1.2.3-alpha1+42"));
    assert_eq!(parse("1.2.3+23"), parse("1.2.3+42"));
}

#[test]
fn test_ne() {
    assert!(parse("0.0.0")       != parse("0.0.1"));
    assert!(parse("0.0.0")       != parse("0.1.0"));
    assert!(parse("0.0.0")       != parse("1.0.0"));
    assert!(parse("1.2.3-alpha") != parse("1.2.3-beta"));
}

#[test]
fn test_show() {
    assert_eq!(format_strbuf!("{}", parse("1.2.3").unwrap()),
               "1.2.3".to_strbuf());
    assert_eq!(format_strbuf!("{}", parse("1.2.3-alpha1").unwrap()),
               "1.2.3-alpha1".to_strbuf());
    assert_eq!(format_strbuf!("{}", parse("1.2.3+build.42").unwrap()),
               "1.2.3+build.42".to_strbuf());
    assert_eq!(format_strbuf!("{}", parse("1.2.3-alpha1+42").unwrap()),
               "1.2.3-alpha1+42".to_strbuf());
}

#[test]
fn test_to_str() {
    assert_eq!(parse("1.2.3").unwrap().to_str(), "1.2.3".to_owned());
    assert_eq!(parse("1.2.3-alpha1").unwrap().to_str(), "1.2.3-alpha1".to_owned());
    assert_eq!(parse("1.2.3+build.42").unwrap().to_str(), "1.2.3+build.42".to_owned());
    assert_eq!(parse("1.2.3-alpha1+42").unwrap().to_str(), "1.2.3-alpha1+42".to_owned());
}

#[test]
fn test_lt() {
    assert!(parse("0.0.0")          < parse("1.2.3-alpha2"));
    assert!(parse("1.0.0")          < parse("1.2.3-alpha2"));
    assert!(parse("1.2.0")          < parse("1.2.3-alpha2"));
    assert!(parse("1.2.3-alpha1")   < parse("1.2.3"));
    assert!(parse("1.2.3-alpha1")   < parse("1.2.3-alpha2"));
    assert!(!(parse("1.2.3-alpha2") < parse("1.2.3-alpha2")));
    assert!(!(parse("1.2.3+23")     < parse("1.2.3+42")));
}

#[test]
fn test_le() {
    assert!(parse("0.0.0")        <= parse("1.2.3-alpha2"));
    assert!(parse("1.0.0")        <= parse("1.2.3-alpha2"));
    assert!(parse("1.2.0")        <= parse("1.2.3-alpha2"));
    assert!(parse("1.2.3-alpha1") <= parse("1.2.3-alpha2"));
    assert!(parse("1.2.3-alpha2") <= parse("1.2.3-alpha2"));
    assert!(parse("1.2.3+23")     <= parse("1.2.3+42"));
}

#[test]
fn test_gt() {
    assert!(parse("1.2.3-alpha2")   > parse("0.0.0"));
    assert!(parse("1.2.3-alpha2")   > parse("1.0.0"));
    assert!(parse("1.2.3-alpha2")   > parse("1.2.0"));
    assert!(parse("1.2.3-alpha2")   > parse("1.2.3-alpha1"));
    assert!(parse("1.2.3")          > parse("1.2.3-alpha2"));
    assert!(!(parse("1.2.3-alpha2") > parse("1.2.3-alpha2")));
    assert!(!(parse("1.2.3+23")     > parse("1.2.3+42")));
}

#[test]
fn test_ge() {
    assert!(parse("1.2.3-alpha2") >= parse("0.0.0"));
    assert!(parse("1.2.3-alpha2") >= parse("1.0.0"));
    assert!(parse("1.2.3-alpha2") >= parse("1.2.0"));
    assert!(parse("1.2.3-alpha2") >= parse("1.2.3-alpha1"));
    assert!(parse("1.2.3-alpha2") >= parse("1.2.3-alpha2"));
    assert!(parse("1.2.3+23")     >= parse("1.2.3+42"));
}

#[test]
fn test_spec_order() {
    let vs = ["1.0.0-alpha",
              "1.0.0-alpha.1",
              "1.0.0-alpha.beta",
              "1.0.0-beta",
              "1.0.0-beta.2",
              "1.0.0-beta.11",
              "1.0.0-rc.1",
              "1.0.0"];
    let mut i = 1;
    while i < vs.len() {
        let a = parse(vs[i-1]).unwrap();
        let b = parse(vs[i]).unwrap();
        assert!(a < b);
        i += 1;
    }
}
