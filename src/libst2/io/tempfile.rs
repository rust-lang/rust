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

use io::{fs, IoResult};
use io;
use libc;
use ops::Drop;
use option::{Option, None, Some};
use os;
use path::{Path, GenericPath};
use result::{Ok, Err};
use sync::atomic;

/// A wrapper for a path to temporary directory implementing automatic
/// scope-based deletion.
pub struct TempDir {
    path: Option<Path>,
    disarmed: bool
}

impl TempDir {
    /// Attempts to make a temporary directory inside of `tmpdir` whose name
    /// will have the suffix `suffix`. The directory will be automatically
    /// deleted once the returned wrapper is destroyed.
    ///
    /// If no directory can be created, `Err` is returned.
    pub fn new_in(tmpdir: &Path, suffix: &str) -> IoResult<TempDir> { unimplemented!() }

    /// Attempts to make a temporary directory inside of `os::tmpdir()` whose
    /// name will have the suffix `suffix`. The directory will be automatically
    /// deleted once the returned wrapper is destroyed.
    ///
    /// If no directory can be created, `Err` is returned.
    pub fn new(suffix: &str) -> IoResult<TempDir> { unimplemented!() }

    /// Unwrap the wrapped `std::path::Path` from the `TempDir` wrapper.
    /// This discards the wrapper so that the automatic deletion of the
    /// temporary directory is prevented.
    pub fn unwrap(self) -> Path { unimplemented!() }

    /// Access the wrapped `std::path::Path` to the temporary directory.
    pub fn path<'a>(&'a self) -> &'a Path { unimplemented!() }

    /// Close and remove the temporary directory
    ///
    /// Although `TempDir` removes the directory on drop, in the destructor
    /// any errors are ignored. To detect errors cleaning up the temporary
    /// directory, call `close` instead.
    pub fn close(mut self) -> IoResult<()> { unimplemented!() }

    fn cleanup_dir(&mut self) -> IoResult<()> { unimplemented!() }
}

impl Drop for TempDir {
    fn drop(&mut self) { unimplemented!() }
}

// the tests for this module need to change the path using change_dir,
// and this doesn't play nicely with other tests so these unit tests are located
// in src/test/run-pass/tempfile.rs
