// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test can't be a unit test in std,
// because it needs TempDir, which is in extra

// ignore-android
// pretty-expanded FIXME #23616

#![feature(rustc_private, path_ext)]

extern crate rustc_back;

use std::ffi::CString;
use std::fs::{self, File, PathExt};
use rustc_back::tempdir::TempDir;

fn rename_directory() {
    let tmpdir = TempDir::new("rename_directory").ok().expect("rename_directory failed");
    let tmpdir = tmpdir.path();
    let old_path = tmpdir.join("foo/bar/baz");
    fs::create_dir_all(&old_path).unwrap();
    let test_file = &old_path.join("temp.txt");

    File::create(test_file).unwrap();

    let new_path = tmpdir.join("quux/blat");
    fs::create_dir_all(&new_path).unwrap();
    fs::rename(&old_path, &new_path.join("newdir"));
    assert!(new_path.join("newdir").is_dir());
    assert!(new_path.join("newdir/temp.txt").exists());
}

pub fn main() { rename_directory() }
