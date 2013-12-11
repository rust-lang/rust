// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Clone, Eq)]
pub struct PkgId {
    path: ~str,
    name: ~str,
    version: Option<~str>,
}

impl ToStr for PkgId {
    fn to_str(&self) -> ~str {
        let version = match self.version {
            None => "0.0",
            Some(ref version) => version.as_slice(),
        };
        if self.path.is_empty() {
            format!("{}\\#{}", self.name, version)
        } else {
            format!("{}/{}\\#{}", self.path, self.name, version)
        }
    }
}

impl FromStr for PkgId {
    fn from_str(s: &str) -> Option<PkgId> {
        let hash_idx = match s.find('#') {
            None => s.len(),
            Some(idx) => idx,
        };
        let prefix = s.slice_to(hash_idx);
        let name_idx = match prefix.rfind('/') {
            None => 0,
            Some(idx) => idx + 1,
        };
        if name_idx >= prefix.len() {
            return None;
        }
        let name = prefix.slice_from(name_idx);
        if name.len() <= 0 {
            return None;
        }

        let path = if name_idx == 0 {
            ""
        } else {
            prefix.slice_to(name_idx - 1)
        };
        let check_path = Path::new(path);
        if !check_path.is_relative() {
            return None;
        }

        let version = match s.find('#') {
            None => None,
            Some(idx) => {
                if idx >= s.len() {
                    None
                } else {
                    let v = s.slice_from(idx + 1);
                    if v.is_empty() {
                        None
                    } else {
                        Some(v.to_owned())
                    }
                }
            }
        };

        Some(PkgId{
            path: path.to_owned(),
            name: name.to_owned(),
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
    assert_eq!(pkgid.path, ~"");
}

#[test]
fn bare_name_single_char() {
    let pkgid: PkgId = from_str("f").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"f");
    assert_eq!(pkgid.version, None);
    assert_eq!(pkgid.path, ~"");
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
    assert_eq!(pkgid.path, ~"example.com/foo");
}

#[test]
fn simple_version() {
    let pkgid: PkgId = from_str("foo#1.0").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"foo");
    assert_eq!(pkgid.version, Some(~"1.0"));
    assert_eq!(pkgid.path, ~"");
}

#[test]
fn absolute_path() {
    let pkgid: Option<PkgId> = from_str("/foo/bar");
    assert!(pkgid.is_none());
}

#[test]
fn path_and_version() {
    let pkgid: PkgId = from_str("example.com/foo/bar#1.0").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"bar");
    assert_eq!(pkgid.version, Some(~"1.0"));
    assert_eq!(pkgid.path, ~"example.com/foo");
}

#[test]
fn single_chars() {
    let pkgid: PkgId = from_str("a/b#1").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"b");
    assert_eq!(pkgid.version, Some(~"1"));
    assert_eq!(pkgid.path, ~"a");
}

#[test]
fn missing_version() {
    let pkgid: PkgId = from_str("foo#").expect("valid pkgid");
    assert_eq!(pkgid.name, ~"foo");
    assert_eq!(pkgid.version, None);
    assert_eq!(pkgid.path, ~"");
}