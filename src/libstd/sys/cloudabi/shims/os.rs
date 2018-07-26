// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use error::Error as StdError;
use ffi::{OsStr, OsString};
use fmt;
use io;
use iter;
use path::{self, PathBuf};
use sys::{unsupported, Void};

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    unsupported()
}

pub type Env = iter::Empty<(OsString, OsString)>;

pub fn env() -> Env {
    iter::empty()
}

pub fn getenv(_: &OsStr) -> io::Result<Option<OsString>> {
    Ok(None)
}

pub fn setenv(_: &OsStr, _: &OsStr) -> io::Result<()> {
    unsupported()
}

pub fn unsetenv(_: &OsStr) -> io::Result<()> {
    unsupported()
}

pub struct SplitPaths<'a>(&'a Void);

pub fn split_paths(_unparsed: &OsStr) -> SplitPaths {
    panic!("unsupported")
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        match *self.0 {}
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(_paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    Err(JoinPathsError)
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "not supported on CloudABI yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    fn description(&self) -> &str {
        "not supported on CloudABI yet"
    }
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn temp_dir() -> PathBuf {
    PathBuf::from("/tmp")
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

pub fn getpid() -> u32 {
    1
}
