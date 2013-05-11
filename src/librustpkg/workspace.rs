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

use path_util::{rust_path, workspace_contains_package_id};
use util::PkgId;
use core::path::Path;

pub fn pkg_parent_workspaces(pkgid: &PkgId, action: &fn(&Path) -> bool) -> bool {
    // Using the RUST_PATH, find workspaces that contain
    // this package ID
    let workspaces = rust_path().filtered(|ws|
        workspace_contains_package_id(pkgid, ws));
    if workspaces.is_empty() {
        // tjc: make this a condition
        fail!("Package %s not found in any of \
                    the following workspaces: %s",
                   pkgid.remote_path.to_str(),
                   rust_path().to_str());
    }
    for workspaces.each |ws| {
        if action(ws) {
            break;
        }
    }
    return true;
}
