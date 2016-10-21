// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of `make clean` in rustbuild.
//!
//! Responsible for cleaning out a build directory of all old and stale
//! artifacts to prepare for a fresh build. Currently doesn't remove the
//! `build/cache` directory (download cache) or the `build/$target/llvm`
//! directory as we want that cached between builds.

use std::fs;
use std::path::Path;

use Build;

pub fn clean(build: &Build) {
    rm_rf(build, "tmp".as_ref());
    rm_rf(build, &build.out.join("tmp"));

    for host in build.config.host.iter() {
        let entries = match build.out.join(host).read_dir() {
            Ok(iter) => iter,
            Err(_) => continue,
        };

        for entry in entries {
            let entry = t!(entry);
            if entry.file_name().to_str() == Some("llvm") {
                continue
            }
            t!(fs::remove_dir_all(&entry.path()));
        }
    }
}

fn rm_rf(build: &Build, path: &Path) {
    if path.exists() {
        build.verbose(&format!("removing `{}`", path.display()));
        t!(fs::remove_dir_all(path));
    }
}
