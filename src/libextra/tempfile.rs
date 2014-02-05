// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Temporary files and directories


use std::os;
use std::rand::Rng;
use std::rand;
use std::io;
use std::io::fs;

/// A wrapper for a path to temporary directory implementing automatic
/// scope-based deletion.
pub struct TempDir {
    priv path: Option<Path>
}

impl TempDir {
    /// Attempts to make a temporary directory inside of `tmpdir` whose name
    /// will have the suffix `suffix`. The directory will be automatically
    /// deleted once the returned wrapper is destroyed.
    ///
    /// If no directory can be created, None is returned.
    pub fn new_in(tmpdir: &Path, suffix: &str) -> Option<TempDir> {
        if !tmpdir.is_absolute() {
            let abs_tmpdir = os::make_absolute(tmpdir);
            return TempDir::new_in(&abs_tmpdir, suffix);
        }

        let mut r = rand::rng();
        for _ in range(0u, 1000) {
            let p = tmpdir.join(r.gen_ascii_str(16) + suffix);
            match fs::mkdir(&p, io::UserRWX) {
                Err(..) => {}
                Ok(()) => return Some(TempDir { path: Some(p) })
            }
        }
        None
    }

    /// Attempts to make a temporary directory inside of `os::tmpdir()` whose
    /// name will have the suffix `suffix`. The directory will be automatically
    /// deleted once the returned wrapper is destroyed.
    ///
    /// If no directory can be created, None is returned.
    pub fn new(suffix: &str) -> Option<TempDir> {
        TempDir::new_in(&os::tmpdir(), suffix)
    }

    /// Unwrap the wrapped `std::path::Path` from the `TempDir` wrapper.
    /// This discards the wrapper so that the automatic deletion of the
    /// temporary directory is prevented.
    pub fn unwrap(self) -> Path {
        let mut tmpdir = self;
        tmpdir.path.take_unwrap()
    }

    /// Access the wrapped `std::path::Path` to the temporary directory.
    pub fn path<'a>(&'a self) -> &'a Path {
        self.path.get_ref()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        for path in self.path.iter() {
            if path.exists() {
                // FIXME: is failing the right thing to do?
                fs::rmdir_recursive(path).unwrap();
            }
        }
    }
}

// the tests for this module need to change the path using change_dir,
// and this doesn't play nicely with other tests so these unit tests are located
// in src/test/run-pass/tempfile.rs
