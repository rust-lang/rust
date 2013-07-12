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
use version::{try_getting_version, Version, NoVersion, split_version};

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

impl Eq for PkgId {
    fn eq(&self, p: &PkgId) -> bool {
        *p.local_path == *self.local_path && p.version == self.version
    }
    fn ne(&self, p: &PkgId) -> bool {
        !(self.eq(p))
    }
}

impl PkgId {
    pub fn new(s: &str) -> PkgId {
        use conditions::bad_pkg_id::cond;

        let mut given_version = None;

        // Did the user request a specific version?
        let s = match split_version(s) {
            Some((path, v)) => {
                debug!("s = %s, path = %s, v = %s", s, path, v.to_str());
                given_version = Some(v);
                path
            }
            None => {
                debug!("%s has no explicit version", s);
                s
            }
        };

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

        let version = match given_version {
            Some(v) => v,
            None => match try_getting_version(&remote_path) {
                Some(v) => v,
                None => NoVersion
            }
        };

        debug!("local_path = %s, remote_path = %s", local_path.to_str(), remote_path.to_str());
        PkgId {
            local_path: local_path,
            remote_path: remote_path,
            short_name: short_name,
            version: version
        }
    }

    pub fn hash(&self) -> ~str {
        fmt!("%s-%s-%s", self.remote_path.to_str(),
             hash(self.remote_path.to_str() + self.version.to_str()),
             self.version.to_str())
    }

    pub fn short_name_with_version(&self) -> ~str {
        fmt!("%s%s", self.short_name, self.version.to_str())
    }
}

impl ToStr for PkgId {
    fn to_str(&self) -> ~str {
        // should probably use the filestem and not the whole path
        fmt!("%s-%s", self.local_path.to_str(), self.version.to_str())
    }
}
