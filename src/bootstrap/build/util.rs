// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::path::{Path, PathBuf};
use std::fs;
use std::process::Command;

use bootstrap::{dylib_path, dylib_path_var};
use filetime::FileTime;

pub fn staticlib(name: &str, target: &str) -> String {
    if target.contains("windows-msvc") {
        format!("{}.lib", name)
    } else {
        format!("lib{}.a", name)
    }
}

pub fn mtime(path: &Path) -> FileTime {
    fs::metadata(path).map(|f| {
        FileTime::from_last_modification_time(&f)
    }).unwrap_or(FileTime::zero())
}

pub fn copy(src: &Path, dst: &Path) {
    let res = fs::hard_link(src, dst);
    let res = res.or_else(|_| fs::copy(src, dst).map(|_| ()));
    if let Err(e) = res {
        panic!("failed to copy `{}` to `{}`: {}", src.display(),
               dst.display(), e)
    }
}

pub fn cp_r(src: &Path, dst: &Path) {
    for f in t!(fs::read_dir(src)) {
        let f = t!(f);
        let path = f.path();
        let name = path.file_name().unwrap();
        let dst = dst.join(name);
        if t!(f.file_type()).is_dir() {
            let _ = fs::remove_dir_all(&dst);
            t!(fs::create_dir(&dst));
            cp_r(&path, &dst);
        } else {
            let _ = fs::remove_file(&dst);
            copy(&path, &dst);
        }
    }
}

/// Given an executable called `name`, return the filename for the
/// executable for a particular target.
pub fn exe(name: &str, target: &str) -> String {
    if target.contains("windows") {
        format!("{}.exe", name)
    } else {
        name.to_string()
    }
}

pub fn is_dylib(name: &str) -> bool {
    name.ends_with(".dylib") || name.ends_with(".so") || name.ends_with(".dll")
}

pub fn libdir(target: &str) -> &'static str {
    if target.contains("windows") {"bin"} else {"lib"}
}

pub fn add_lib_path(path: Vec<PathBuf>, cmd: &mut Command) {
    let mut list = dylib_path();
    for path in path {
        list.insert(0, path);
    }
    cmd.env(dylib_path_var(), t!(env::join_paths(list)));
}

#[allow(dead_code)] // this will be used soon
pub fn up_to_date(src: &Path, dst: &Path) -> bool {
    let threshold = mtime(dst);
    let meta = t!(fs::metadata(src));
    if meta.is_dir() {
        dir_up_to_date(src, &threshold)
    } else {
        FileTime::from_last_modification_time(&meta) <= threshold
    }
}

fn dir_up_to_date(src: &Path, threshold: &FileTime) -> bool {
    t!(fs::read_dir(src)).map(|e| t!(e)).all(|e| {
        let meta = t!(e.metadata());
        if meta.is_dir() {
            dir_up_to_date(&e.path(), threshold)
        } else {
            FileTime::from_last_modification_time(&meta) < *threshold
        }
    })
}
