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

use io::{fs, IoError, IoErrorKind, IoResult};
use io;
use iter::{IteratorExt, range};
use ops::Drop;
use option::Option;
use option::Option::{None, Some};
use os;
use path::{Path, GenericPath};
use rand::{Rng, thread_rng};
use result::Result::{Ok, Err};
use str::StrExt;
use string::String;

/// A wrapper for a path to temporary directory implementing automatic
/// scope-based deletion.
///
/// # Examples
///
/// ```no_run
/// use std::io::TempDir;
///
/// {
///     // create a temporary directory
///     let tmpdir = match TempDir::new("myprefix") {
///         Ok(dir) => dir,
///         Err(e) => panic!("couldn't create temporary directory: {}", e)
///     };
///
///     // get the path of the temporary directory without affecting the wrapper
///     let tmppath = tmpdir.path();
///
///     println!("The path of temporary directory is {}", tmppath.display());
///
///     // the temporary directory is automatically removed when tmpdir goes
///     // out of scope at the end of the block
/// }
/// {
///     // create a temporary directory, this time using a custom path
///     let tmpdir = match TempDir::new_in(&Path::new("/tmp/best/custom/path"), "myprefix") {
///         Ok(dir) => dir,
///         Err(e) => panic!("couldn't create temporary directory: {}", e)
///     };
///
///     // get the path of the temporary directory and disable automatic deletion in the wrapper
///     let tmppath = tmpdir.into_inner();
///
///     println!("The path of the not-so-temporary directory is {}", tmppath.display());
///
///     // the temporary directory is not removed here
///     // because the directory is detached from the wrapper
/// }
/// {
///     // create a temporary directory
///     let tmpdir = match TempDir::new("myprefix") {
///         Ok(dir) => dir,
///         Err(e) => panic!("couldn't create temporary directory: {}", e)
///     };
///
///     // close the temporary directory manually and check the result
///     match tmpdir.close() {
///         Ok(_) => println!("success!"),
///         Err(e) => panic!("couldn't remove temporary directory: {}", e)
///     };
/// }
/// ```
pub struct TempDir {
    path: Option<Path>,
    disarmed: bool
}

// How many times should we (re)try finding an unused random name? It should be
// enough that an attacker will run out of luck before we run out of patience.
const NUM_RETRIES: u32 = 1 << 31;
// How many characters should we include in a random file name? It needs to
// be enough to dissuade an attacker from trying to preemptively create names
// of that length, but not so huge that we unnecessarily drain the random number
// generator of entropy.
const NUM_RAND_CHARS: uint = 12;

impl TempDir {
    /// Attempts to make a temporary directory inside of `tmpdir` whose name
    /// will have the prefix `prefix`. The directory will be automatically
    /// deleted once the returned wrapper is destroyed.
    ///
    /// If no directory can be created, `Err` is returned.
    pub fn new_in(tmpdir: &Path, prefix: &str) -> IoResult<TempDir> {
        if !tmpdir.is_absolute() {
            let abs_tmpdir = try!(os::make_absolute(tmpdir));
            return TempDir::new_in(&abs_tmpdir, prefix);
        }

        let mut rng = thread_rng();
        for _ in range(0, NUM_RETRIES) {
            let suffix: String = rng.gen_ascii_chars().take(NUM_RAND_CHARS).collect();
            let leaf = if prefix.len() > 0 {
                format!("{}.{}", prefix, suffix)
            } else {
                // If we're given an empty string for a prefix, then creating a
                // directory starting with "." would lead to it being
                // semi-invisible on some systems.
                suffix
            };
            let path = tmpdir.join(leaf);
            match fs::mkdir(&path, io::USER_RWX) {
                Ok(_) => return Ok(TempDir { path: Some(path), disarmed: false }),
                Err(IoError{kind:IoErrorKind::PathAlreadyExists,..}) => (),
                Err(e) => return Err(e)
            }
        }

        return Err(IoError{
                       kind: IoErrorKind::PathAlreadyExists,
                       desc:"Exhausted",
                       detail: None});
    }

    /// Attempts to make a temporary directory inside of `os::tmpdir()` whose
    /// name will have the prefix `prefix`. The directory will be automatically
    /// deleted once the returned wrapper is destroyed.
    ///
    /// If no directory can be created, `Err` is returned.
    pub fn new(prefix: &str) -> IoResult<TempDir> {
        TempDir::new_in(&os::tmpdir(), prefix)
    }

    /// Unwrap the wrapped `std::path::Path` from the `TempDir` wrapper.
    /// This discards the wrapper so that the automatic deletion of the
    /// temporary directory is prevented.
    pub fn into_inner(self) -> Path {
        let mut tmpdir = self;
        tmpdir.path.take().unwrap()
    }

    /// Access the wrapped `std::path::Path` to the temporary directory.
    pub fn path<'a>(&'a self) -> &'a Path {
        self.path.as_ref().unwrap()
    }

    /// Close and remove the temporary directory
    ///
    /// Although `TempDir` removes the directory on drop, in the destructor
    /// any errors are ignored. To detect errors cleaning up the temporary
    /// directory, call `close` instead.
    pub fn close(mut self) -> IoResult<()> {
        self.cleanup_dir()
    }

    fn cleanup_dir(&mut self) -> IoResult<()> {
        assert!(!self.disarmed);
        self.disarmed = true;
        match self.path {
            Some(ref p) => {
                fs::rmdir_recursive(p)
            }
            None => Ok(())
        }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if !self.disarmed {
            let _ = self.cleanup_dir();
        }
    }
}

// the tests for this module need to change the path using change_dir,
// and this doesn't play nicely with other tests so these unit tests are located
// in src/test/run-pass/tempfile.rs
