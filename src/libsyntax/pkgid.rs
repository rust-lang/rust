// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// PkgIds identify crates and include the crate name and optionall a path and
/// version. In the full form, they look like relative URLs. Example:
/// `github.com/mozilla/rust#std:1.0` would be a package ID with a path of
/// `gitub.com/mozilla/rust` and a crate name of `std` with a version of
/// `1.0`. If no crate name is given after the hash, the name is inferred to
/// be the last component of the path. If no version is given, it is inferred
/// to be `0.0`.
#[deriving(Clone, Eq)]
pub struct PkgId {
    /// A path which represents the codes origin. By convention this is the
    /// URL, without `http://` or `https://` prefix, to the crate's repository
    path: ~str,
    /// The name of the crate.
    name: ~str,
    /// The version of the crate.
    version: Option<~str>,
}

impl ToStr for PkgId {
    fn to_str(&self) -> ~str {
        let version = match self.version {
            None => "0.0",
            Some(ref version) => version.as_slice(),
        };
        if self.path == self.name || self.path.ends_with(format!("/{}", self.name)) {
            format!("{}\\#{}", self.path, version)
        } else {
            format!("{}\\#{}:{}", self.path, self.name, version)
        }
    }
}

impl FromStr for PkgId {
    fn from_str(s: &str) -> Option<PkgId> {
        let pieces: ~[&str] = s.splitn('#', 1).collect();
        let path = pieces[0].to_owned();

        if path.starts_with("/") || path.ends_with("/") ||
            path.starts_with(".") || path.is_empty() {
            return None;
        }

        let path_pieces: ~[&str] = path.rsplitn('/', 1).collect();
        let inferred_name = path_pieces[0];

        let (name, version) = if pieces.len() == 1 {
            (inferred_name.to_owned(), None)
        } else {
            let hash_pieces: ~[&str] = pieces[1].splitn(':', 1).collect();
            let (hash_name, hash_version) = if hash_pieces.len() == 1 {
                ("", hash_pieces[0])
            } else {
                (hash_pieces[0], hash_pieces[1])
            };

            let name = if !hash_name.is_empty() {
                hash_name.to_owned()
            } else {
                inferred_name.to_owned()
            };

            let version = if !hash_version.is_empty() {
                Some(hash_version.to_owned())
            } else {
                None
            };

            (name, version)
        };

        Some(PkgId {
            path: path,
            name: name,
            version: version,
        })
    }
}

impl PkgId {
    pub fn version_or_default<'a>(&'a self) -> &'a str {
        match self.version {
            None => "0.0",
            Some(ref version) => version.as_slice(),
        }
    }
}

#[test]
fn bare_name() {
    let pkgid: PkgId = from_str("foo").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"foo");
    assert_eq!(pkgid.version, None);
    assert_eq!(pkgid.path, ~"foo");
}

#[test]
fn bare_name_single_char() {
    let pkgid: PkgId = from_str("f").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"f");
    assert_eq!(pkgid.version, None);
    assert_eq!(pkgid.path, ~"f");
}

#[test]
fn empty_pkgid() {
    let pkgid: Option<PkgId> = from_str("");
    assert!(pkgid.is_none());
}

#[test]
fn simple_path() {
    let pkgid: PkgId = from_str("example.com/foo/bar").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"bar");
    assert_eq!(pkgid.version, None);
    assert_eq!(pkgid.path, ~"example.com/foo/bar");
}

#[test]
fn simple_version() {
    let pkgid: PkgId = from_str("foo#1.0").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"foo");
    assert_eq!(pkgid.version, Some(~"1.0"));
    assert_eq!(pkgid.path, ~"foo");
}

#[test]
fn absolute_path() {
    let pkgid: Option<PkgId> = from_str("/foo/bar");
    assert!(pkgid.is_none());
}

#[test]
fn path_ends_with_slash() {
    let pkgid: Option<PkgId> = from_str("foo/bar/");
    assert!(pkgid.is_none());
}

#[test]
fn path_and_version() {
    let pkgid: PkgId = from_str("example.com/foo/bar#1.0").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"bar");
    assert_eq!(pkgid.version, Some(~"1.0"));
    assert_eq!(pkgid.path, ~"example.com/foo/bar");
}

#[test]
fn single_chars() {
    let pkgid: PkgId = from_str("a/b#1").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"b");
    assert_eq!(pkgid.version, Some(~"1"));
    assert_eq!(pkgid.path, ~"a/b");
}

#[test]
fn missing_version() {
    let pkgid: PkgId = from_str("foo#").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"foo");
    assert_eq!(pkgid.version, None);
    assert_eq!(pkgid.path, ~"foo");
}

#[test]
fn path_and_name() {
    let pkgid: PkgId = from_str("foo/rust-bar#bar:1.0").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"bar");
    assert_eq!(pkgid.version, Some(~"1.0"));
    assert_eq!(pkgid.path, ~"foo/rust-bar");
}

#[test]
fn empty_name() {
    let pkgid: PkgId = from_str("foo/bar#:1.0").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"bar");
    assert_eq!(pkgid.version, Some(~"1.0"));
    assert_eq!(pkgid.path, ~"foo/bar");
}
