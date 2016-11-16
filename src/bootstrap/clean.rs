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
use std::io::{self, ErrorKind};
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
            let path = t!(entry.path().canonicalize());
            rm_rf(build, &path);
        }
    }
}

fn rm_rf(build: &Build, path: &Path) {
    if !path.exists() {
        return
    }
    if path.is_file() {
        return do_op(path, "remove file", |p| fs::remove_file(p));
    }

    for file in t!(fs::read_dir(path)) {
        let file = t!(file).path();

        if file.is_dir() {
            rm_rf(build, &file);
        } else {
            // On windows we can't remove a readonly file, and git will
            // often clone files as readonly. As a result, we have some
            // special logic to remove readonly files on windows.
            do_op(&file, "remove file", |p| fs::remove_file(p));
        }
    }
    do_op(path, "remove dir", |p| fs::remove_dir(p));
}

fn do_op<F>(path: &Path, desc: &str, mut f: F)
    where F: FnMut(&Path) -> io::Result<()>
{
    match f(path) {
        Ok(()) => {}
        Err(ref e) if cfg!(windows) &&
                      e.kind() == ErrorKind::PermissionDenied => {
            let mut p = t!(path.metadata()).permissions();
            p.set_readonly(false);
            t!(fs::set_permissions(path, p));
            f(path).unwrap_or_else(|e| {
                panic!("failed to {} {}: {}", desc, path.display(), e);
            })
        }
        Err(e) => {
            panic!("failed to {} {}: {}", desc, path.display(), e);
        }
    }
}
