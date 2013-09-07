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
use std::{char, os, result, run, str};
use extra::tempfile::mkdtemp;
use path_util::rust_path;

#[deriving(Clone)]
pub enum Version {
    ExactRevision(~str), // Should look like a m.n.(...).x
    SemanticVersion(semver::Version),
    Tagged(~str), // String that can't be parsed as a version.
                  // Requirements get interpreted exactly
    NoVersion // user didn't specify a version -- prints as 0.1
}

// Equality on versions is non-symmetric: if self is NoVersion, it's equal to
// anything; but if self is a precise version, it's not equal to NoVersion.
// We should probably make equality symmetric, and use less-than and greater-than
// where we currently use eq
impl Eq for Version {
    fn eq(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(ref s1), &ExactRevision(ref s2)) => *s1 == *s2,
            (&SemanticVersion(ref v1), &SemanticVersion(ref v2)) => *v1 == *v2,
            (&NoVersion, _) => true,
            _ => false
        }
    }
}

impl Ord for Version {
    fn lt(&self, other: &Version) -> bool {
        match (self, other) {
            (&NoVersion, _) => true,
            (&ExactRevision(ref f1), &ExactRevision(ref f2)) => f1 < f2,
            (&SemanticVersion(ref v1), &SemanticVersion(ref v2)) => v1 < v2,
            _ => false // incomparable, really
        }
    }
    fn le(&self, other: &Version) -> bool {
        match (self, other) {
            (&NoVersion, _) => true,
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
            ExactRevision(ref n) | Tagged(ref n) => fmt!("%s", n.to_str()),
            SemanticVersion(ref v) => fmt!("%s", v.to_str()),
            NoVersion => ~"0.1"
        }
    }
}

pub fn parse_vers(vers: ~str) -> result::Result<semver::Version, ~str> {
    match semver::parse(vers) {
        Some(vers) => result::Ok(vers),
        None => result::Err(~"could not parse version: invalid")
    }
}

/// If `local_path` is a git repo in the RUST_PATH, and the most recent tag
/// in that repo denotes a version, return it; otherwise, `None`
pub fn try_getting_local_version(local_path: &Path) -> Option<Version> {
    let rustpath = rust_path();
    for rp in rustpath.iter() {
        let local_path = rp.push_rel(local_path);
        let git_dir = local_path.push(".git");
        if !os::path_is_dir(&git_dir) {
            loop;
        }
        let outp = run::process_output("git",
                                   [fmt!("--git-dir=%s", git_dir.to_str()), ~"tag", ~"-l"]);

        debug!("git --git-dir=%s tag -l ~~~> %?", git_dir.to_str(), outp.status);

        if outp.status != 0 {
            loop;
        }

    let mut output = None;
    let output_text = str::from_utf8(outp.output);
    for l in output_text.line_iter() {
        if !l.is_whitespace() {
            output = Some(l);
        }
        match output.chain(try_parsing_version) {
            Some(v) => return Some(v),
            None    => ()
        }
    }
  }
  None
}

/// If `remote_path` refers to a git repo that can be downloaded,
/// and the most recent tag in that repo denotes a version, return it;
/// otherwise, `None`
pub fn try_getting_version(remote_path: &Path) -> Option<Version> {
    if is_url_like(remote_path) {
        let tmp_dir = mkdtemp(&os::tmpdir(),
                              "test").expect("try_getting_version: couldn't create temp dir");
        debug!("(to get version) executing {git clone https://%s %s}",
               remote_path.to_str(),
               tmp_dir.to_str());
        let outp  = run::process_output("git", [~"clone", fmt!("https://%s", remote_path.to_str()),
                                                tmp_dir.to_str()]);
        if outp.status == 0 {
            debug!("Cloned it... ( %s, %s )",
                   str::from_utf8(outp.output),
                   str::from_utf8(outp.error));
            let mut output = None;
            debug!("(getting version, now getting tags) executing {git --git-dir=%s tag -l}",
                   tmp_dir.push(".git").to_str());
            let outp = run::process_output("git",
                                           [fmt!("--git-dir=%s", tmp_dir.push(".git").to_str()),
                                            ~"tag", ~"-l"]);
            let output_text = str::from_utf8(outp.output);
            debug!("Full output: ( %s ) [%?]", output_text, outp.status);
            for l in output_text.line_iter() {
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

pub fn try_parsing_version(s: &str) -> Option<Version> {
    let s = s.trim();
    debug!("Attempting to parse: %s", s);
    let mut parse_state = Start;
    for c in s.iter() {
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
fn is_url_like(p: &Path) -> bool {
    let str = p.to_str();
    str.split_iter('/').len() > 2
}

/// If s is of the form foo#bar, where bar is a valid version
/// number, return the prefix before the # and the version.
/// Otherwise, return None.
pub fn split_version<'a>(s: &'a str) -> Option<(&'a str, Version)> {
    // Check for extra '#' characters separately
    if s.split_iter('#').len() > 2 {
        return None;
    }
    split_version_general(s, '#')
}

pub fn split_version_general<'a>(s: &'a str, sep: char) -> Option<(&'a str, Version)> {
    match s.rfind(sep) {
        Some(i) => {
            let path = s.slice(0, i);
            // n.b. for now, assuming an exact revision is intended, not a SemVer
            Some((path, ExactRevision(s.slice(i + 1, s.len()).to_owned())))
        }
        None => {
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
