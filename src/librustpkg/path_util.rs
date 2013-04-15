// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustpkg utilities having to do with paths and directories

use core::path::*;
use core::os;
use util::PkgId;

/// Returns the output directory to use.
/// Right now is always the default, should
/// support changing it.
pub fn dest_dir(pkgid: PkgId) -> Path {
    default_dest_dir(&pkgid.path)
}

/// Returns the default output directory for compilation.
/// Creates that directory if it doesn't exist.
pub fn default_dest_dir(pkg_dir: &Path) -> Path {
    use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
    use conditions::bad_path::cond;

    // For now: assumes that pkg_dir exists and is relative
    // to the CWD. Change this later when we do path searching.
    let rslt = pkg_dir.push("build");
    let is_dir = os::path_is_dir(&rslt);
    if os::path_exists(&rslt) {
        if is_dir {
            rslt
        }
        else {
            cond.raise((rslt, ~"Path names a file that isn't a directory"))
        }
    }
    else {
        // Create it
        if os::make_dir(&rslt, (S_IRUSR | S_IWUSR | S_IXUSR) as i32) {
            rslt
        }
        else {
            cond.raise((rslt, ~"Could not create directory"))
        }
    }
}

#[cfg(test)]
mod test {
    use core::{os, rand};
    use core::path::Path;
    use core::rand::RngUtil;
    use path_util::*;

    // Helper function to create a directory name that doesn't exist
    pub fn mk_nonexistent(tmpdir: &Path, suffix: &str) -> Path {
        let r = rand::Rng();
        for 1000.times {
            let p = tmpdir.push(r.gen_str(16) + suffix);
            if !os::path_exists(&p) {
                return p;
            }
        }
        fail!(~"Couldn't compute a non-existent path name; this is worrisome")
    }

    #[test]
    fn default_dir_ok() {
        let the_path = os::tmpdir();
        let substitute_path = Path("xyzzy");
        assert!(default_dest_dir(&the_path) == the_path.push(~"build"));
        let nonexistent_path = mk_nonexistent(&the_path, "quux");
        let bogus = do ::conditions::bad_path::cond.trap(|_| {
            substitute_path
        }).in { default_dest_dir(&nonexistent_path) };
        assert!(bogus == substitute_path);
    }
}
