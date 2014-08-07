// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
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
/// `github.com/rust-lang/rust#std:1.0` would be a package ID with a path of
/// `github.com/rust-lang/rust` and a crate name of `std` with a version of
/// `1.0`. If no crate name is given after the hash, the name is inferred to
/// be the last component of the path. If no version is given, it is inferred
/// to be `0.0`.

use std::from_str::FromStr;

#[deriving(Clone, PartialEq)]
pub struct CrateId {
    /// A path which represents the codes origin. By convention this is the
    /// URL, without `http://` or `https://` prefix, to the crate's repository
    pub path: String,
    /// The name of the crate.
    pub name: String,
    /// The version of the crate.
    pub version: Option<String>,
}

impl fmt::Show for CrateId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{}", self.path));
        let version = match self.version {
            None => "0.0",
            Some(ref version) => version.as_slice(),
        };
        if self.path == self.name ||
                self.path
                    .as_slice()
                    .ends_with(format!("/{}", self.name).as_slice()) {
            write!(f, "#{}", version)
        } else {
            write!(f, "#{}:{}", self.name, version)
        }
    }
}

impl FromStr for CrateId {
    fn from_str(s: &str) -> Option<CrateId> {
        let pieces: Vec<&str> = s.splitn(1, '#').collect();
        let path = pieces.get(0).to_string();

        if path.as_slice().starts_with("/") || path.as_slice().ends_with("/") ||
            path.as_slice().starts_with(".") || path.is_empty() {
            return None;
        }

        let path_pieces: Vec<&str> = path.as_slice()
                                         .rsplitn(1, '/')
                                         .collect();
        let inferred_name = *path_pieces.get(0);

        let (name, version) = if pieces.len() == 1 {
            (inferred_name.to_string(), None)
        } else {
            let hash_pieces: Vec<&str> = pieces.get(1)
                                               .splitn(1, ':')
                                               .collect();
            let (hash_name, hash_version) = if hash_pieces.len() == 1 {
                ("", *hash_pieces.get(0))
            } else {
                (*hash_pieces.get(0), *hash_pieces.get(1))
            };

            let name = if !hash_name.is_empty() {
                hash_name.to_string()
            } else {
                inferred_name.to_string()
            };

            let version = if !hash_version.is_empty() {
                if hash_version == "0.0" {
                    None
                } else {
                    Some(hash_version.to_string())
                }
            } else {
                None
            };

            (name, version)
        };

        Some(CrateId {
            path: path.to_string(),
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

    pub fn short_name_with_version(&self) -> String {
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
    assert_eq!(crateid.name, "foo".to_string());
    assert_eq!(crateid.version, None);
    assert_eq!(crateid.path, "foo".to_string());
}

#[test]
fn bare_name_single_char() {
    let crateid: CrateId = from_str("f").expect("valid crateid");
    assert_eq!(crateid.name, "f".to_string());
    assert_eq!(crateid.version, None);
    assert_eq!(crateid.path, "f".to_string());
}

#[test]
fn empty_crateid() {
    let crateid: Option<CrateId> = from_str("");
    assert!(crateid.is_none());
}

#[test]
fn simple_path() {
    let crateid: CrateId = from_str("example.com/foo/bar").expect("valid crateid");
    assert_eq!(crateid.name, "bar".to_string());
    assert_eq!(crateid.version, None);
    assert_eq!(crateid.path, "example.com/foo/bar".to_string());
}

#[test]
fn simple_version() {
    let crateid: CrateId = from_str("foo#1.0").expect("valid crateid");
    assert_eq!(crateid.name, "foo".to_string());
    assert_eq!(crateid.version, Some("1.0".to_string()));
    assert_eq!(crateid.path, "foo".to_string());
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
    assert_eq!(crateid.name, "bar".to_string());
    assert_eq!(crateid.version, Some("1.0".to_string()));
    assert_eq!(crateid.path, "example.com/foo/bar".to_string());
}

#[test]
fn single_chars() {
    let crateid: CrateId = from_str("a/b#1").expect("valid crateid");
    assert_eq!(crateid.name, "b".to_string());
    assert_eq!(crateid.version, Some("1".to_string()));
    assert_eq!(crateid.path, "a/b".to_string());
}

#[test]
fn missing_version() {
    let crateid: CrateId = from_str("foo#").expect("valid crateid");
    assert_eq!(crateid.name, "foo".to_string());
    assert_eq!(crateid.version, None);
    assert_eq!(crateid.path, "foo".to_string());
}

#[test]
fn path_and_name() {
    let crateid: CrateId = from_str("foo/rust-bar#bar:1.0").expect("valid crateid");
    assert_eq!(crateid.name, "bar".to_string());
    assert_eq!(crateid.version, Some("1.0".to_string()));
    assert_eq!(crateid.path, "foo/rust-bar".to_string());
}

#[test]
fn empty_name() {
    let crateid: CrateId = from_str("foo/bar#:1.0").expect("valid crateid");
    assert_eq!(crateid.name, "bar".to_string());
    assert_eq!(crateid.version, Some("1.0".to_string()));
    assert_eq!(crateid.path, "foo/bar".to_string());
}
