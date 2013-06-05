// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use package_path::{RemotePath, LocalPath, normalize, hash};
use extra::semver;
use core::prelude::*;
use core::result;

/// Placeholder
pub fn default_version() -> Version { ExactRevision(0.1) }

/// Path-fragment identifier of a package such as
/// 'github.com/graydon/test'; path must be a relative
/// path with >=1 component.
pub struct PkgId {
    /// Remote path: for example, github.com/mozilla/quux-whatever
    remote_path: RemotePath,
    /// Local path: for example, /home/quux/github.com/mozilla/quux_whatever
    /// Note that '-' normalizes to '_' when mapping a remote path
    /// onto a local path
    /// Also, this will change when we implement #6407, though we'll still
    /// need to keep track of separate local and remote paths
    local_path: LocalPath,
    /// Short name. This is the local path's filestem, but we store it
    /// redundantly so as to not call get() everywhere (filestem() returns an
    /// option)
    short_name: ~str,
    version: Version
}

pub impl PkgId {
    fn new(s: &str) -> PkgId {
        use conditions::bad_pkg_id::cond;

        let p = Path(s);
        if p.is_absolute {
            return cond.raise((p, ~"absolute pkgid"));
        }
        if p.components.len() < 1 {
            return cond.raise((p, ~"0-length pkgid"));
        }
        let remote_path = RemotePath(p);
        let local_path = normalize(copy remote_path);
        let short_name = (copy local_path).filestem().expect(fmt!("Strange path! %s", s));
        PkgId {
            local_path: local_path,
            remote_path: remote_path,
            short_name: short_name,
            version: default_version()
        }
    }

    fn hash(&self) -> ~str {
        fmt!("%s-%s-%s", self.remote_path.to_str(),
             hash(self.remote_path.to_str() + self.version.to_str()),
             self.version.to_str())
    }

    fn short_name_with_version(&self) -> ~str {
        fmt!("%s-%s", self.short_name, self.version.to_str())
    }
}

impl ToStr for PkgId {
    fn to_str(&self) -> ~str {
        // should probably use the filestem and not the whole path
        fmt!("%s-%s", self.local_path.to_str(), self.version.to_str())
    }
}

/// A version is either an exact revision,
/// or a semantic version
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
