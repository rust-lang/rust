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
use std::io::{self, Error, ErrorKind};
use std::fs;
use std::path::{self, PathBuf, Path};
use std::__rand::{thread_rng, Rng};

/// A wrapper for a path to temporary directory implementing automatic
/// scope-based deletion.
pub struct TempDir {
    path: Option<PathBuf>,
}

// How many times should we (re)try finding an unused random name? It should be
// enough that an attacker will run out of luck before we run out of patience.
const NUM_RETRIES: u32 = 1 << 31;
// How many characters should we include in a random file name? It needs to
// be enough to dissuade an attacker from trying to preemptively create names
// of that length, but not so huge that we unnecessarily drain the random number
// generator of entropy.
const NUM_RAND_CHARS: usize = 12;

impl TempDir {
    /// Attempts to make a temporary directory inside of `tmpdir` whose name
    /// will have the prefix `prefix`. The directory will be automatically
    /// deleted once the returned wrapper is destroyed.
    ///
    /// If no directory can be created, `Err` is returned.
    #[allow(deprecated)] // rand usage
    pub fn new_in<P: AsRef<Path>>(tmpdir: P, prefix: &str)
                                  -> io::Result<TempDir> {
        let storage;
        let mut tmpdir = tmpdir.as_ref();
        if !tmpdir.is_absolute() {
            let cur_dir = try!(env::current_dir());
            storage = cur_dir.join(tmpdir);
            tmpdir = &storage;
            // return TempDir::new_in(&cur_dir.join(tmpdir), prefix);
        }

        let mut rng = thread_rng();
        for _ in 0..NUM_RETRIES {
            let suffix: String = rng.gen_ascii_chars().take(NUM_RAND_CHARS).collect();
            let leaf = if !prefix.is_empty() {
                format!("{}.{}", prefix, suffix)
            } else {
                // If we're given an empty string for a prefix, then creating a
                // directory starting with "." would lead to it being
                // semi-invisible on some systems.
                suffix
            };
            let path = tmpdir.join(&leaf);
            match fs::create_dir(&path) {
                Ok(_) => return Ok(TempDir { path: Some(path) }),
                Err(ref e) if e.kind() == ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e)
            }
        }

        Err(Error::new(ErrorKind::AlreadyExists,
                       "too many temporary directories already exist"))
    }

    /// Attempts to make a temporary directory inside of `env::temp_dir()` whose
    /// name will have the prefix `prefix`. The directory will be automatically
    /// deleted once the returned wrapper is destroyed.
    ///
    /// If no directory can be created, `Err` is returned.
    #[allow(deprecated)]
    pub fn new(prefix: &str) -> io::Result<TempDir> {
        TempDir::new_in(&env::temp_dir(), prefix)
    }

    /// Unwrap the wrapped `std::path::Path` from the `TempDir` wrapper.
    /// This discards the wrapper so that the automatic deletion of the
    /// temporary directory is prevented.
    pub fn into_path(mut self) -> PathBuf {
        self.path.take().unwrap()
    }

    /// Access the wrapped `std::path::Path` to the temporary directory.
    pub fn path(&self) -> &path::Path {
        self.path.as_ref().unwrap()
    }

    /// Close and remove the temporary directory
    ///
    /// Although `TempDir` removes the directory on drop, in the destructor
    /// any errors are ignored. To detect errors cleaning up the temporary
    /// directory, call `close` instead.
    pub fn close(mut self) -> io::Result<()> {
        self.cleanup_dir()
    }

    fn cleanup_dir(&mut self) -> io::Result<()> {
        match self.path {
            Some(ref p) => fs::remove_dir_all(p),
            None => Ok(())
        }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = self.cleanup_dir();
    }
}

// the tests for this module need to change the path using change_dir,
// and this doesn't play nicely with other tests so these unit tests are located
// in src/test/run-pass/tempfile.rs
