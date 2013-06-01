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

use extra::semver;
use core::prelude::*;
use core::{char, os, result, run, str};
use package_path::RemotePath;
use extra::tempfile::mkdtemp;

#[deriving(Eq)]
pub enum Version {
    ExactRevision(~str), // Should look like a m.n.(...).x
    SemanticVersion(semver::Version),
    NoVersion // user didn't specify a version
}


impl Ord for Version {
    fn lt(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(ref f1), &ExactRevision(ref f2)) => f1 < f2,
            (&SemanticVersion(ref v1), &SemanticVersion(ref v2)) => v1 < v2,
            _ => false // incomparable, really
        }
    }
    fn le(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(ref f1), &ExactRevision(ref f2)) => f1 <= f2,
            (&SemanticVersion(ref v1), &SemanticVersion(ref v2)) => v1 <= v2,
            _ => false // incomparable, really
        }
    }
    fn ge(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(ref f1), &ExactRevision(ref f2)) => f1 > f2,
            (&SemanticVersion(ref v1), &SemanticVersion(ref v2)) => v1 > v2,
            _ => false // incomparable, really
        }
    }
    fn gt(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(ref f1), &ExactRevision(ref f2)) => f1 >= f2,
            (&SemanticVersion(ref v1), &SemanticVersion(ref v2)) => v1 >= v2,
            _ => false // incomparable, really
        }
    }

}

impl ToStr for Version {
    fn to_str(&self) -> ~str {
        match *self {
            ExactRevision(ref n) => fmt!("%s", n.to_str()),
            SemanticVersion(ref v) => fmt!("%s", v.to_str()),
            NoVersion => ~""
        }
    }
}

impl Version {
    /// Fills in a bogus default version for NoVersion -- for use when
    /// injecting link_meta attributes
    fn to_str_nonempty(&self) -> ~str {
        match *self {
            NoVersion => ~"0.1",
            _ => self.to_str()
        }
    }
}


pub fn parse_vers(vers: ~str) -> result::Result<semver::Version, ~str> {
    match semver::parse(vers) {
        Some(vers) => result::Ok(vers),
        None => result::Err(~"could not parse version: invalid")
    }
}


/// If `remote_path` refers to a git repo that can be downloaded,
/// and the most recent tag in that repo denotes a version, return it;
/// otherwise, `None`
pub fn try_getting_version(remote_path: &RemotePath) -> Option<Version> {
    debug!("try_getting_version: %s", remote_path.to_str());
    if is_url_like(remote_path) {
        debug!("Trying to fetch its sources..");
        let tmp_dir = mkdtemp(&os::tmpdir(),
                              "test").expect("try_getting_version: couldn't create temp dir");
        debug!("executing {git clone https://%s %s}", remote_path.to_str(), tmp_dir.to_str());
        let outp  = run::process_output("git", [~"clone", fmt!("https://%s", remote_path.to_str()),
                                                tmp_dir.to_str()]);
        if outp.status == 0 {
            debug!("Cloned it... ( %s, %s )",
                   str::from_bytes(outp.output),
                   str::from_bytes(outp.error));
            let mut output = None;
            debug!("executing {git --git-dir=%s tag -l}", tmp_dir.push(".git").to_str());
            let outp = run::process_output("git",
                                           [fmt!("--git-dir=%s", tmp_dir.push(".git").to_str()),
                                            ~"tag", ~"-l"]);
            let output_text = str::from_bytes(outp.output);
            debug!("Full output: ( %s ) [%?]", output_text, outp.status);
            for output_text.each_split_char('\n') |l| {
                debug!("A line of output: %s", l);
                if !l.is_whitespace() {
                    output = Some(l);
                }
            }

            output.chain(try_parsing_version)
        }
        else {
            None
        }
    }
    else {
        None
    }
}

// Being lazy since we don't have a regexp library now
#[deriving(Eq)]
enum ParseState {
    Start,
    SawDigit,
    SawDot
}

fn try_parsing_version(s: &str) -> Option<Version> {
    let s = s.trim();
    debug!("Attempting to parse: %s", s);
    let mut parse_state = Start;
    // I gave up on using external iterators (tjc)
    for str::to_chars(s).each() |&c| {
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
        SawDigit => Some(ExactRevision(s.to_owned())),
        _        => None
    }
}

/// Just an approximation
fn is_url_like(p: &RemotePath) -> bool {
    let mut n = 0;
    for p.to_str().each_split_char('/') |_| {
        n += 1;
    }
    n > 2
}

/// If s is of the form foo#bar, where bar is a valid version
/// number, return the prefix before the # and the version.
/// Otherwise, return None.
pub fn split_version<'a>(s: &'a str) -> Option<(&'a str, Version)> {
    // reject strings with multiple '#'s
    if { let mut i: uint = 0; for str::to_chars(s).each |&c| { if c == '#' { i += 1; } }; i > 1 } {
        return None;
    }
    match str::rfind_char(s, '#') {
        Some(i) => {
            debug!("in %s, i = %?", s, i);
            let path = s.slice(0, i);
            debug!("path = %s", path);
            // n.b. for now, assuming an exact revision is intended, not a SemVer
            Some((path, ExactRevision(s.slice(i + 1, s.len()).to_owned())))
        }
        None => {
            debug!("%s doesn't look like an explicit-version thing", s);
            None
        }
    }
}

#[test]
fn test_parse_version() {
    assert!(try_parsing_version("1.2") == Some(ExactRevision(~"1.2")));
    assert!(try_parsing_version("1.0.17") == Some(ExactRevision(~"1.0.17")));
    assert!(try_parsing_version("you're_a_kitty") == None);
    assert!(try_parsing_version("42..1") == None);
    assert!(try_parsing_version("17") == Some(ExactRevision(~"17")));
    assert!(try_parsing_version(".1.2.3") == None);
    assert!(try_parsing_version("2.3.") == None);
}

#[test]
fn test_split_version() {
    let s = "a/b/c#0.1";
    debug!("== %? ==", split_version(s));
    assert!(split_version(s) == Some((s.slice(0, 5), ExactRevision(~"0.1"))));
    assert!(split_version("a/b/c") == None);
    let s = "a#1.2";
    assert!(split_version(s) == Some((s.slice(0, 1), ExactRevision(~"1.2"))));
    assert!(split_version("a#a#3.4") == None);
}