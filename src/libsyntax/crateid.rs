// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

/// CrateIds identify crates and include the crate name and optionally a path
/// and version. In the full form, they look like relative URLs. Example:
/// `github.com/mozilla/rust#std:1.0` would be a package ID with a path of
/// `gitub.com/mozilla/rust` and a crate name of `std` with a version of
/// `1.0`. If no crate name is given after the hash, the name is inferred to
/// be the last component of the path. If no version is given, it is inferred
/// to be `0.0`.

use std::from_str::FromStr;

#[deriving(Clone, Eq)]
pub struct CrateId {
    /// A path which represents the codes origin. By convention this is the
    /// URL, without `http://` or `https://` prefix, to the crate's repository
    pub path: ~str,
    /// The name of the crate.
    pub name: ~str,
    /// The version of the crate.
    pub version: Option<~str>,
}

impl fmt::Show for CrateId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f.buf, "{}", self.path));
        let version = match self.version {
            None => "0.0",
            Some(ref version) => version.as_slice(),
        };
        if self.path == self.name || self.path.ends_with(format!("/{}", self.name)) {
            write!(f.buf, "\\#{}", version)
        } else {
            write!(f.buf, "\\#{}:{}", self.name, version)
        }
    }
}

impl FromStr for CrateId {
    fn from_str(s: &str) -> Option<CrateId> {
        let pieces: Vec<&str> = s.splitn('#', 1).collect();
        let path = pieces.get(0).to_owned();

        if path.starts_with("/") || path.ends_with("/") ||
            path.starts_with(".") || path.is_empty() {
            return None;
        }

        let path_pieces: Vec<&str> = path.rsplitn('/', 1).collect();
        let inferred_name = *path_pieces.get(0);

        let (name, version) = if pieces.len() == 1 {
            (inferred_name.to_owned(), None)
        } else {
            let hash_pieces: Vec<&str> = pieces.get(1)
                                               .splitn(':', 1)
                                               .collect();
            let (hash_name, hash_version) = if hash_pieces.len() == 1 {
                ("", *hash_pieces.get(0))
            } else {
                (*hash_pieces.get(0), *hash_pieces.get(1))
            };

            let name = if !hash_name.is_empty() {
                hash_name.to_owned()
            } else {
                inferred_name.to_owned()
            };

            let version = if !hash_version.is_empty() {
                if hash_version == "0.0" {
                    None
                } else {
                    Some(hash_version.to_owned())
                }
            } else {
                None
            };

            (name, version)
        };

        Some(CrateId {
            path: path.clone(),
            name: name,
            version: version,
        })
    }
}

impl CrateId {
    pub fn version_or_default<'a>(&'a self) -> &'a str {
        match self.version {
            None => "0.0",
            Some(ref version) => version.as_slice(),
        }
    }

    pub fn short_name_with_version(&self) -> ~str {
        format!("{}-{}", self.name, self.version_or_default())
    }

    pub fn matches(&self, other: &CrateId) -> bool {
        // FIXME: why does this not match on `path`?
        if self.name != other.name { return false }
        match (&self.version, &other.version) {
            (&Some(ref v1), &Some(ref v2)) => v1 == v2,
            _ => true,
        }
    }
}

#[test]
fn bare_name() {
    let crateid: CrateId = from_str("foo").expect("valid crateid");
    assert_eq!(crateid.name, ~"foo");
    assert_eq!(crateid.version, None);
    assert_eq!(crateid.path, ~"foo");
}

#[test]
fn bare_name_single_char() {
    let crateid: CrateId = from_str("f").expect("valid crateid");
    assert_eq!(crateid.name, ~"f");
    assert_eq!(crateid.version, None);
    assert_eq!(crateid.path, ~"f");
}

#[test]
fn empty_crateid() {
    let crateid: Option<CrateId> = from_str("");
    assert!(crateid.is_none());
}

#[test]
fn simple_path() {
    let crateid: CrateId = from_str("example.com/foo/bar").expect("valid crateid");
    assert_eq!(crateid.name, ~"bar");
    assert_eq!(crateid.version, None);
    assert_eq!(crateid.path, ~"example.com/foo/bar");
}

#[test]
fn simple_version() {
    let crateid: CrateId = from_str("foo#1.0").expect("valid crateid");
    assert_eq!(crateid.name, ~"foo");
    assert_eq!(crateid.version, Some(~"1.0"));
    assert_eq!(crateid.path, ~"foo");
}

#[test]
fn absolute_path() {
    let crateid: Option<CrateId> = from_str("/foo/bar");
    assert!(crateid.is_none());
}

#[test]
fn path_ends_with_slash() {
    let crateid: Option<CrateId> = from_str("foo/bar/");
    assert!(crateid.is_none());
}

#[test]
fn path_and_version() {
    let crateid: CrateId = from_str("example.com/foo/bar#1.0").expect("valid crateid");
    assert_eq!(crateid.name, ~"bar");
    assert_eq!(crateid.version, Some(~"1.0"));
    assert_eq!(crateid.path, ~"example.com/foo/bar");
}

#[test]
fn single_chars() {
    let crateid: CrateId = from_str("a/b#1").expect("valid crateid");
    assert_eq!(crateid.name, ~"b");
    assert_eq!(crateid.version, Some(~"1"));
    assert_eq!(crateid.path, ~"a/b");
}

#[test]
fn missing_version() {
    let crateid: CrateId = from_str("foo#").expect("valid crateid");
    assert_eq!(crateid.name, ~"foo");
    assert_eq!(crateid.version, None);
    assert_eq!(crateid.path, ~"foo");
}

#[test]
fn path_and_name() {
    let crateid: CrateId = from_str("foo/rust-bar#bar:1.0").expect("valid crateid");
    assert_eq!(crateid.name, ~"bar");
    assert_eq!(crateid.version, Some(~"1.0"));
    assert_eq!(crateid.path, ~"foo/rust-bar");
}

#[test]
fn empty_name() {
    let crateid: CrateId = from_str("foo/bar#:1.0").expect("valid crateid");
    assert_eq!(crateid.name, ~"bar");
    assert_eq!(crateid.version, Some(~"1.0"));
    assert_eq!(crateid.path, ~"foo/bar");
}
