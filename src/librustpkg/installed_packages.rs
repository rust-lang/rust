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

pub fn list_installed_packages(f: &fn(&PkgId) -> bool) -> bool  {
    let workspaces = rust_path();
    for p in workspaces.iter() {
        let binfiles = os::list_dir(&p.push("bin"));
        for exec in binfiles.iter() {
            let p = Path(*exec);
            let exec_path = p.filestem();
            do exec_path.iter().advance |s| {
                f(&PkgId::new(*s))
            };
        }
        let libfiles = os::list_dir(&p.push("lib"));
        for lib in libfiles.iter() {
            let lib = Path(*lib);
            debug2!("Full name: {}", lib.to_str());
            match has_library(&lib) {
                Some(basename) => {
                    debug2!("parent = {}, child = {}",
                            p.push("lib").to_str(), lib.to_str());
                    let rel_p = p.push("lib/").get_relative_to(&lib);
                    debug2!("Rel: {}", rel_p.to_str());
                    let rel_path = rel_p.push(basename).to_str();
                    debug2!("Rel name: {}", rel_path);
                    f(&PkgId::new(rel_path));
                }
                None => ()
            }
        };
    }
    true
}

pub fn has_library(p: &Path) -> Option<~str> {
    let files = os::list_dir(p);
    for q in files.iter() {
        let as_path = Path(*q);
        if as_path.filetype() == Some(os::consts::DLL_SUFFIX) {
            let stuff : &str = as_path.filestem().expect("has_library: weird path");
            let mut stuff2 = stuff.split_str_iter(&"-");
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
        }
        false
    };
    is_installed
}
