// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use path_util::{installed_library_in_workspace, rust_path};
use version::Version;

/// If some workspace `p` in the RUST_PATH contains a package matching short_name,
/// return Some(p) (returns the first one of there are multiple matches.) Return
/// None if there's no such path.
/// FIXME #8711: This ignores the desired version.
pub fn find_installed_library_in_rust_path(pkg_path: &Path, _version: &Version) -> Option<Path> {
    let rp = rust_path();
    debug!("find_installed_library_in_rust_path: looking for path %s", pkg_path.to_str());
    for p in rp.iter() {
        match installed_library_in_workspace(pkg_path, p) {
            Some(path) => return Some(path),
            None => ()
        }
    }
    None
}
