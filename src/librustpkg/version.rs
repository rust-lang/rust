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

use std::semver;
use core::prelude::*;
use core::run;
use package_path::RemotePath;
use std::tempfile::mkdtemp;

pub enum Version {
    ExactRevision(float),
    SemVersion(semver::Version)
}


impl Ord for Version {
    fn lt(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 < f2,
            (&SemVersion(ref v1), &SemVersion(ref v2)) => v1 < v2,
            _ => false // incomparable, really
        }
    }
    fn le(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 <= f2,
            (&SemVersion(ref v1), &SemVersion(ref v2)) => v1 <= v2,
            _ => false // incomparable, really
        }
    }
    fn ge(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 > f2,
            (&SemVersion(ref v1), &SemVersion(ref v2)) => v1 > v2,
            _ => false // incomparable, really
        }
    }
    fn gt(&self, other: &Version) -> bool {
        match (self, other) {
            (&ExactRevision(f1), &ExactRevision(f2)) => f1 >= f2,
            (&SemVersion(ref v1), &SemVersion(ref v2)) => v1 >= v2,
            _ => false // incomparable, really
        }
    }

}

impl ToStr for Version {
    fn to_str(&self) -> ~str {
        match *self {
            ExactRevision(ref n) => n.to_str(),
            SemVersion(ref v) => v.to_str()
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
            debug!("Cloned it... ( %s, %s )", str::from_bytes(outp.output), str::from_bytes(outp.error));
            let mut output = None;
            debug!("executing {git --git-dir=%s tag -l}", tmp_dir.push(".git").to_str());
            let outp = run::process_output("git", [fmt!("--git-dir=%s", tmp_dir.push(".git").to_str()),
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
    
fn try_parsing_version(s: &str) -> Option<Version> {
    let s = s.trim();
    debug!("Attempting to parse: %s", s);
    match float::from_str(s) {
        Some(f) => {
            debug!("%s -> %f", s, f);
            Some(ExactRevision(f)) // semver not handled yet
        }
        None => {
            debug!("None!!");
            None
        }
    }
}

/// Placeholder
pub fn default_version() -> Version { ExactRevision(0.1) }

/// Just an approximation
fn is_url_like(p: &RemotePath) -> bool {
    let mut n = 0;
    for p.to_str().each_split_char('/') |_| {
        n += 1;
    }
    n > 2
}