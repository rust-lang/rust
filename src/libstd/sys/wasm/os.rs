// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use error::Error as StdError;
use ffi::{OsString, OsStr};
use fmt;
use io;
use path::{self, PathBuf};
use str;
use sys::{unsupported, Void, ExitSysCall, GetEnvSysCall, SetEnvSysCall};

pub fn errno() -> i32 {
    0
}

pub fn error_string(_errno: i32) -> String {
    format!("operation successful")
}

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
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
    where I: Iterator<Item=T>, T: AsRef<OsStr>
{
    Err(JoinPathsError)
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "not supported on wasm yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    fn description(&self) -> &str {
        "not supported on wasm yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

pub struct Env(Void);

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        match self.0 {}
    }
}

pub fn env() -> Env {
    panic!("not supported on web assembly")
}

pub fn getenv(k: &OsStr) -> io::Result<Option<OsString>> {
    Ok(GetEnvSysCall::perform(k))
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    Ok(SetEnvSysCall::perform(k, Some(v)))
}

pub fn unsetenv(k: &OsStr) -> io::Result<()> {
    Ok(SetEnvSysCall::perform(k, None))
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on wasm")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(_code: i32) -> ! {
    ExitSysCall::perform(_code as isize as usize)
}

pub fn getpid() -> u32 {
    panic!("no pids on wasm")
}
