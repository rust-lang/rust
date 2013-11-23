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

use rustc::metadata::filesearch::rust_path;
use path_util::*;
use std::os;
use std::io;
use std::io::fs;

pub fn list_installed_packages(f: &fn(&PkgId) -> bool) -> bool  {
    let workspaces = rust_path();
    for p in workspaces.iter() {
        let binfiles = do io::ignore_io_error { fs::readdir(&p.join("bin")) };
        for exec in binfiles.iter() {
            // FIXME (#9639): This needs to handle non-utf8 paths
            match exec.filestem_str() {
                None => (),
                Some(exec_path) => {
                    if !f(&PkgId::new(exec_path)) {
                        return false;
                    }
                }
            }
        }
        let libfiles = do io::ignore_io_error { fs::readdir(&p.join("lib")) };
        for lib in libfiles.iter() {
            debug!("Full name: {}", lib.display());
            match has_library(lib) {
                Some(basename) => {
                    let parent = p.join("lib");
                    debug!("parent = {}, child = {}",
                            parent.display(), lib.display());
                    let rel_p = lib.path_relative_from(&parent).unwrap();
                    debug!("Rel: {}", rel_p.display());
                    let rel_path = rel_p.join(basename);
                    do rel_path.display().with_str |s| {
                        debug!("Rel name: {}", s);
                        f(&PkgId::new(s));
                    }
                }
                None => ()
            }
        };
    }
    true
}

pub fn has_library(p: &Path) -> Option<~str> {
    let files = do io::ignore_io_error { fs::readdir(p) };
    for path in files.iter() {
        if path.extension_str() == Some(os::consts::DLL_EXTENSION) {
            let stuff : &str = path.filestem_str().expect("has_library: weird path");
            let mut stuff2 = stuff.split_str("-");
            let stuff3: ~[&str] = stuff2.collect();
            // argh
            let chars_to_drop = os::consts::DLL_PREFIX.len();
            return Some(stuff3[0].slice(chars_to_drop, stuff3[0].len()).to_owned());
        }
    }
    None
}

pub fn package_is_installed(p: &PkgId) -> bool {
    let mut is_installed = false;
    do list_installed_packages() |installed| {
        if installed == p {
            is_installed = true;
            false
        } else {
            true
        }
    };
    is_installed
}
