// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fortanix_sgx_abi::{Error, RESULT_SUCCESS};

use error::Error as StdError;
use ffi::{OsString, OsStr};
use fmt;
use io;
use path::{self, PathBuf};
use str;
use sys::{unsupported, Void, sgx_ineffective, decode_error_kind};

pub fn errno() -> i32 {
    RESULT_SUCCESS
}

pub fn error_string(errno: i32) -> String {
    if errno == RESULT_SUCCESS {
        "operation succesful".into()
    } else if ((Error::UserRangeStart as _)..=(Error::UserRangeEnd as _)).contains(&errno) {
        format!("user-specified error {:08x}", errno)
    } else {
        decode_error_kind(errno).as_str().into()
    }
}

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    sgx_ineffective(())
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
        "not supported in SGX yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    fn description(&self) -> &str {
        "not supported in SGX yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

pub struct Env;

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        None
    }
}

pub fn env() -> Env {
    Env
}

pub fn getenv(_k: &OsStr) -> io::Result<Option<OsString>> {
    Ok(None)
}

pub fn setenv(_k: &OsStr, _v: &OsStr) -> io::Result<()> {
    sgx_ineffective(()) // FIXME: this could trigger a panic higher up the stack
}

pub fn unsetenv(_k: &OsStr) -> io::Result<()> {
    sgx_ineffective(()) // FIXME: this could trigger a panic higher up the stack
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem in SGX")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    super::abi::exit_with_code(code as _)
}

pub fn getpid() -> u32 {
    panic!("no pids in SGX")
}
