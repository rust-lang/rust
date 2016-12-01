// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of `std::os` functionality for unix systems

#![allow(unused_imports)] // lots of cfg code here

use os::unix::prelude::*;

use error::Error as StdError;
use ffi::{OsString, OsStr};
use fmt;
use io::{self, Read, Write};
use iter;
use marker::PhantomData;
use mem;
use memchr;
use path::{self, PathBuf};
use ptr;
use slice;
use str;
use sys_common::mutex::Mutex;
use sys::{cvt, fd, syscall};
use vec;

const TMPBUF_SZ: usize = 128;
static ENV_LOCK: Mutex = Mutex::new();

/// Returns the platform-specific value of errno
pub fn errno() -> i32 {
    0
}

/// Gets a detailed string description for the given error number.
pub fn error_string(errno: i32) -> String {
    if let Some(string) = syscall::STR_ERROR.get(errno as usize) {
        string.to_string()
    } else {
        "unknown error".to_string()
    }
}

pub fn getcwd() -> io::Result<PathBuf> {
    let mut buf = [0; 4096];
    let count = cvt(syscall::getcwd(&mut buf))?;
    Ok(PathBuf::from(OsString::from_vec(buf[.. count].to_vec())))
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    cvt(syscall::chdir(p.to_str().unwrap())).and(Ok(()))
}

pub struct SplitPaths<'a> {
    iter: iter::Map<slice::Split<'a, u8, fn(&u8) -> bool>,
                    fn(&'a [u8]) -> PathBuf>,
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths {
    fn bytes_to_path(b: &[u8]) -> PathBuf {
        PathBuf::from(<OsStr as OsStrExt>::from_bytes(b))
    }
    fn is_colon(b: &u8) -> bool { *b == b':' }
    let unparsed = unparsed.as_bytes();
    SplitPaths {
        iter: unparsed.split(is_colon as fn(&u8) -> bool)
                      .map(bytes_to_path as fn(&[u8]) -> PathBuf)
    }
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
    where I: Iterator<Item=T>, T: AsRef<OsStr>
{
    let mut joined = Vec::new();
    let sep = b':';

    for (i, path) in paths.enumerate() {
        let path = path.as_ref().as_bytes();
        if i > 0 { joined.push(sep) }
        if path.contains(&sep) {
            return Err(JoinPathsError)
        }
        joined.extend_from_slice(path);
    }
    Ok(OsStringExt::from_vec(joined))
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "path segment contains separator `:`".fmt(f)
    }
}

impl StdError for JoinPathsError {
    fn description(&self) -> &str { "failed to join paths" }
}

pub fn current_exe() -> io::Result<PathBuf> {
    use fs::File;

    let mut file = File::open("sys:exe")?;

    let mut path = String::new();
    file.read_to_string(&mut path)?;

    if path.ends_with('\n') {
        path.pop();
    }

    Ok(PathBuf::from(path))
}

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env() -> Env {
    let mut variables: Vec<(OsString, OsString)> = Vec::new();
    if let Ok(mut file) = ::fs::File::open("env:") {
        let mut string = String::new();
        if file.read_to_string(&mut string).is_ok() {
            for line in string.lines() {
                let mut parts = line.splitn(2, '=');
                if let Some(name) = parts.next() {
                    let value = parts.next().unwrap_or("");
                    variables.push((OsString::from(name.to_string()),
                                    OsString::from(value.to_string())));
                }
            }
        }
    }
    Env { iter: variables.into_iter(), _dont_send_or_sync_me: PhantomData }
}

pub fn getenv(key: &OsStr) -> io::Result<Option<OsString>> {
    if ! key.is_empty() {
        if let Ok(mut file) = ::fs::File::open(&("env:".to_owned() + key.to_str().unwrap())) {
            let mut string = String::new();
            file.read_to_string(&mut string)?;
            Ok(Some(OsString::from(string)))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

pub fn setenv(key: &OsStr, value: &OsStr) -> io::Result<()> {
    if ! key.is_empty() {
        let mut file = ::fs::File::open(&("env:".to_owned() + key.to_str().unwrap()))?;
        file.write_all(value.as_bytes())?;
        file.set_len(value.len() as u64)?;
    }
    Ok(())
}

pub fn unsetenv(key: &OsStr) -> io::Result<()> {
    ::fs::remove_file(&("env:".to_owned() + key.to_str().unwrap()))?;
    Ok(())
}

pub fn page_size() -> usize {
    4096
}

pub fn temp_dir() -> PathBuf {
    ::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/tmp")
    })
}

pub fn home_dir() -> Option<PathBuf> {
    return ::env::var_os("HOME").map(PathBuf::from);
}

pub fn exit(code: i32) -> ! {
    let _ = syscall::exit(code as usize);
    unreachable!();
}
