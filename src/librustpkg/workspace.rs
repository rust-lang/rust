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
use context::Context;
use path_util::{workspace_contains_crate_id, find_dir_using_rust_path_hack, default_workspace};
use path_util::rust_path;
use util::option_to_vec;
use crate_id::CrateId;

pub fn each_pkg_parent_workspace(cx: &Context,
                                 crateid: &CrateId,
                                 action: |&Path| -> bool)
                                 -> bool {
    // Using the RUST_PATH, find workspaces that contain
    // this package ID
    let workspaces = pkg_parent_workspaces(cx, crateid);
    if workspaces.is_empty() {
        // tjc: make this a condition
        fail!("Package {} not found in any of \
                    the following workspaces: {}",
                   crateid.path.display(),
                   rust_path().map(|p| p.display().to_str()).to_str());
    }
    for ws in workspaces.iter() {
        if action(ws) {
            break;
        }
    }
    return true;
}

/// Given a package ID, return a vector of all of the workspaces in
/// the RUST_PATH that contain it
pub fn pkg_parent_workspaces(cx: &Context, crateid: &CrateId) -> ~[Path] {
    let rs: ~[Path] = rust_path().move_iter()
        .filter(|ws| workspace_contains_crate_id(crateid, ws))
        .collect();
    if cx.use_rust_path_hack {
        rs + option_to_vec(find_dir_using_rust_path_hack(crateid))
    }
    else {
        rs
    }
}

/// Construct a workspace and package-ID name based on the current directory.
/// This gets used when rustpkg gets invoked without a package-ID argument.
pub fn cwd_to_workspace() -> Option<(Path, CrateId)> {
    let cwd = os::getcwd();
    for path in rust_path().move_iter() {
        let srcpath = path.join("src");
        if srcpath.is_ancestor_of(&cwd) {
            let rel = cwd.path_relative_from(&srcpath);
            let rel_s = rel.as_ref().and_then(|p|p.as_str());
            if rel_s.is_some() {
                return Some((path, CrateId::new(rel_s.unwrap())));
            }
        }
    }
    None
}

/// If `workspace` is the same as `cwd`, and use_rust_path_hack is false,
/// return `workspace`; otherwise, return the first workspace in the RUST_PATH.
pub fn determine_destination(cwd: Path, use_rust_path_hack: bool, workspace: &Path) -> Path {
    if workspace == &cwd && !use_rust_path_hack {
        workspace.clone()
    }
    else {
        default_workspace()
    }
}
