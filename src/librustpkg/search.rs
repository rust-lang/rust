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

/// If a library with path `p` matching pkg_id's name exists under sroot_opt,
/// return Some(p). Return None if there's no such path or if sroot_opt is None.
pub fn find_library_in_search_path(sroot_opt: Option<@Path>, short_name: &str) -> Option<Path> {
    do sroot_opt.chain |sroot| {
        debug!("Will search for a library with short name %s in \
                %s", short_name, (sroot.push("lib")).to_str());
        installed_library_in_workspace(short_name, sroot)
    }
}

/// If some workspace `p` in the RUST_PATH contains a package matching short_name,
/// return Some(p) (returns the first one of there are multiple matches.) Return
/// None if there's no such path.
/// FIXME #8711: This ignores the desired version.
pub fn find_installed_library_in_rust_path(short_name: &str, _version: &Version) -> Option<Path> {
    let rp = rust_path();
    for p in rp.iter() {
        match installed_library_in_workspace(short_name, p) {
            Some(path) => return Some(path),
            None => ()
        }
    }
    None
}
