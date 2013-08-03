// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustpkg utilities having to do with workspaces

use std::os;
use std::path::Path;
use path_util::{rust_path, workspace_contains_package_id};
use package_id::PkgId;

pub fn each_pkg_parent_workspace(pkgid: &PkgId, action: &fn(&Path) -> bool) -> bool {
    // Using the RUST_PATH, find workspaces that contain
    // this package ID
    let workspaces = pkg_parent_workspaces(pkgid);
    if workspaces.is_empty() {
        // tjc: make this a condition
        fail!("Package %s not found in any of \
                    the following workspaces: %s",
                   pkgid.remote_path.to_str(),
                   rust_path().to_str());
    }
    for ws in workspaces.iter() {
        if action(ws) {
            break;
        }
    }
    return true;
}

pub fn pkg_parent_workspaces(pkgid: &PkgId) -> ~[Path] {
    rust_path().consume_iter()
        .filter(|ws| workspace_contains_package_id(pkgid, ws))
        .collect()
}

pub fn in_workspace(complain: &fn()) -> bool {
    let dir_part = os::getcwd().pop().components.clone();
    if  *(dir_part.last()) != ~"src" {
        complain();
        false
    }
    else {
        true
    }
}

/// Construct a workspace and package-ID name based on the current directory.
/// This gets used when rustpkg gets invoked without a package-ID argument.
pub fn cwd_to_workspace() -> (Path, PkgId) {
    let cwd = os::getcwd();
    let ws = cwd.pop().pop();
    let cwd_ = cwd.clone();
    let pkgid = cwd_.components.last().to_str();
    (ws, PkgId::new(pkgid, &cwd))
}
