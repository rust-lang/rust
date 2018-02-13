// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::intrinsics;

use error::Error as StdError;
use ffi::{OsString, OsStr};
use fmt;
use io;
use mem;
use path::{self, PathBuf};
use str;
use sys::{unsupported, Void};

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
    // If we're debugging the runtime then we actually probe node.js to ask for
    // the value of environment variables to help provide inputs to programs.
    // The `extern` shims here are defined in `src/etc/wasm32-shim.js` and are
    // intended for debugging only, you should not rely on them.
    if !super::DEBUG {
        return Ok(None)
    }

    extern {
        fn rust_wasm_getenv_len(k: *const u8, kl: usize) -> isize;
        fn rust_wasm_getenv_data(k: *const u8, kl: usize, v: *mut u8);
    }
    unsafe {
        let k: &[u8] = mem::transmute(k);
        let n = rust_wasm_getenv_len(k.as_ptr(), k.len());
        if n == -1 {
            return Ok(None)
        }
        let mut data = vec![0; n as usize];
        rust_wasm_getenv_data(k.as_ptr(), k.len(), data.as_mut_ptr());
        Ok(Some(mem::transmute(data)))
    }
}

pub fn setenv(_k: &OsStr, _v: &OsStr) -> io::Result<()> {
    unsupported()
}

pub fn unsetenv(_n: &OsStr) -> io::Result<()> {
    unsupported()
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on wasm")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(_code: i32) -> ! {
    unsafe { intrinsics::abort() }
}

pub fn getpid() -> u32 {
    panic!("no pids on wasm")
}
