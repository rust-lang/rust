// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Listing installed packages

use path_util::*;
use std::os;

pub fn list_installed_packages(f: &fn(&PkgId) -> bool) -> bool  {
    let workspaces = rust_path();
    for workspaces.iter().advance |p| {
        let binfiles = os::list_dir(&p.push("bin"));
        for binfiles.iter().advance() |exec| {
            f(&PkgId::new(*exec, p));
        }
        let libfiles = os::list_dir(&p.push("lib"));
        for libfiles.iter().advance() |lib| {
            f(&PkgId::new(*lib, p));
        }
    }
    true
}

pub fn package_is_installed(p: &PkgId) -> bool {
    let mut is_installed = false;
    do list_installed_packages() |installed| {
        if installed == p {
            is_installed = true;
        }
        false
    };
    is_installed
}