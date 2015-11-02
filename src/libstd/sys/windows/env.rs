// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of `std::os` functionality for Windows

#![allow(bad_style)]

use os::windows::prelude::*;

use borrow::Cow;
use ffi::{OsString, OsStr};
use libc::types::os::arch::extra::LPWCH;
use libc::{self, c_int, c_void};
use ops::Range;
use os::windows::ffi::EncodeWide;
use ptr;
use result;
use slice;
use sys::inner::*;
use sys::windows::c;
use sys::windows::handle::Handle;
use sys::error::{self, Result};
use vec::Vec;

use libc::funcs::extra::kernel32::{
    GetEnvironmentStringsW,
    FreeEnvironmentStringsW
};

pub use sys::common::env::{JoinPathsError, ARCH};

fn os2path(s: &[u16]) -> OsString {
    OsString::from_wide(s)
}

fn to_utf16_os(s: &OsStr) -> Vec<u16> {
    let mut v: Vec<_> = s.encode_wide().collect();
    v.push(0);
    v
}

pub struct Vars {
    base: LPWCH,
    cur: LPWCH,
}

impl Iterator for Vars {
    type Item = (OsString, OsString);

    fn next(&mut self) -> Option<(OsString, OsString)> {
        unsafe {
            if *self.cur == 0 { return None }
            let p = &*self.cur;
            let mut len = 0;
            while *(p as *const u16).offset(len) != 0 {
                len += 1;
            }
            let p = p as *const u16;
            let s = slice::from_raw_parts(p, len as usize);
            self.cur = self.cur.offset(len + 1);

            let (k, v) = match s.iter().position(|&b| b == '=' as u16) {
                Some(n) => (&s[..n], &s[n+1..]),
                None => (s, &[][..]),
            };
            Some((OsStringExt::from_wide(k), OsStringExt::from_wide(v)))
        }
    }
}

impl Drop for Vars {
    fn drop(&mut self) {
        unsafe { FreeEnvironmentStringsW(self.base); }
    }
}

pub fn vars() -> Result<Vars> {
    unsafe {
        c::cvt(GetEnvironmentStringsW() as usize).map(|ch| Vars { base: ch as *mut _, cur: ch as *mut _ })
    }
}

pub struct SplitPaths<'a> {
    data: EncodeWide<'a>,
    must_yield: bool,
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths {
    SplitPaths {
        data: unparsed.encode_wide(),
        must_yield: true,
    }
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = Cow<'a, OsStr>;

    fn next(&mut self) -> Option<Cow<'a, OsStr>> {
        // On Windows, the PATH environment variable is semicolon separated.
        // Double quotes are used as a way of introducing literal semicolons
        // (since c:\some;dir is a valid Windows path). Double quotes are not
        // themselves permitted in path names, so there is no way to escape a
        // double quote.  Quoted regions can appear in arbitrary locations, so
        //
        //   c:\foo;c:\som"e;di"r;c:\bar
        //
        // Should parse as [c:\foo, c:\some;dir, c:\bar].
        //
        // (The above is based on testing; there is no clear reference available
        // for the grammar.)


        let must_yield = self.must_yield;
        self.must_yield = false;

        let mut in_progress = Vec::new();
        let mut in_quote = false;
        for b in self.data.by_ref() {
            if b == '"' as u16 {
                in_quote = !in_quote;
            } else if b == ';' as u16 && !in_quote {
                self.must_yield = true;
                break
            } else {
                in_progress.push(b)
            }
        }

        if !must_yield && in_progress.is_empty() {
            None
        } else {
            Some(Cow::Owned(os2path(&in_progress)))
        }
    }
}

pub fn join_paths<I, T>(paths: I) -> result::Result<OsString, JoinPathsError>
    where I: Iterator<Item=T>, T: AsRef<OsStr>
{
    let mut joined = Vec::new();
    let sep = b';' as u16;

    for (i, path) in paths.enumerate() {
        let path = path.as_ref();
        if i > 0 { joined.push(sep) }
        let v = path.encode_wide().collect::<Vec<u16>>();
        if v.contains(&(b'"' as u16)) {
            return Err(JoinPathsError::new())
        } else if v.contains(&sep) {
            joined.push(b'"' as u16);
            joined.push_all(&v[..]);
            joined.push(b'"' as u16);
        } else {
            joined.push_all(&v[..]);
        }
    }

    Ok(OsStringExt::from_wide(&joined[..]))
}

pub fn join_paths_error() -> &'static str {
    "path segment contains `\"`"
}

pub fn current_exe() -> Result<OsString> {
    c::fill_utf16_buf(|buf, sz| unsafe {
        libc::GetModuleFileNameW(ptr::null_mut(), buf, sz)
    }, os2path)
}

pub fn getcwd() -> Result<OsString> {
    c::fill_utf16_buf(|buf, sz| unsafe {
        libc::GetCurrentDirectoryW(sz, buf)
    }, os2path)
}

pub fn chdir(p: &OsStr) -> Result<()> {
    let mut p = p.encode_wide().collect::<Vec<_>>();
    p.push(0);

    unsafe {
        match libc::SetCurrentDirectoryW(p.as_ptr()) != (0 as libc::BOOL) {
            true => Ok(()),
            false => error::expect_last_result(),
        }
    }
}

pub fn getenv(k: &OsStr) -> Result<Option<OsString>> {
    let k = to_utf16_os(k);
    Ok(c::fill_utf16_buf(|buf, sz| unsafe {
        libc::GetEnvironmentVariableW(k.as_ptr(), buf, sz)
    }, |buf| {
        OsStringExt::from_wide(buf)
    }).ok())
}

pub fn setenv(k: &OsStr, v: &OsStr) -> Result<()> {
    let k = to_utf16_os(k);
    let v = to_utf16_os(v);

    unsafe {
        c::cvt(libc::SetEnvironmentVariableW(k.as_ptr(), v.as_ptr())).map(drop)
    }
}

pub fn unsetenv(n: &OsStr) -> Result<()>  {
    let v = to_utf16_os(n);
    unsafe {
        c::cvt(libc::SetEnvironmentVariableW(v.as_ptr(), ptr::null())).map(drop)
    }
}

pub struct Args {
    range: Range<isize>,
    cur: *mut *mut u16,
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> {
        self.range.next().map(|i| unsafe {
            let ptr = *self.cur.offset(i);
            let mut len = 0;
            while *ptr.offset(len) != 0 { len += 1; }

            // Push it onto the list.
            let ptr = ptr as *const u16;
            let buf = slice::from_raw_parts(ptr, len as usize);
            OsStringExt::from_wide(buf)
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) { self.range.size_hint() }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize { self.range.len() }
}

impl Drop for Args {
    fn drop(&mut self) {
        // self.cur can be null if CommandLineToArgvW previously failed,
        // but LocalFree ignores NULL pointers
        unsafe { c::LocalFree(self.cur as *mut c_void); }
    }
}

pub fn args() -> Result<Args> {
    unsafe {
        let mut nArgs: c_int = 0;
        let lpCmdLine = c::GetCommandLineW();
        let szArgList = try!(c::cvt(c::CommandLineToArgvW(lpCmdLine, &mut nArgs) as usize)) as *mut _;

        // szArcList can be NULL if CommandLinToArgvW failed,
        // but in that case nArgs is 0 so we won't actually
        // try to read a null pointer
        Ok(Args { cur: szArgList, range: 0..(nArgs as isize) })
    }
}

pub fn temp_dir() -> Result<OsString> {
    c::fill_utf16_buf(|buf, sz| unsafe {
        c::GetTempPathW(sz, buf)
    }, os2path)
}

pub fn home_dir() -> Result<OsString> {
    getenv("HOME".as_ref()).unwrap_or(None).or_else(|| {
        getenv("USERPROFILE".as_ref()).unwrap_or(None)
    }).map_or_else(|| unsafe {
        let me = c::GetCurrentProcess();
        let mut token = ptr::null_mut();
        try!(c::cvt(c::OpenProcessToken(me, c::TOKEN_READ, &mut token)));
        let _handle = Handle::from_inner(token);
        c::fill_utf16_buf(|buf, mut sz| {
            match c::GetUserProfileDirectoryW(token, buf, &mut sz) {
                0 if libc::GetLastError() != 0 => 0,
                0 => sz,
                n => n as libc::DWORD,
            }
        }, os2path)
    }, Ok)
}

pub const FAMILY: &'static str = "windows";
pub const OS: &'static str = "windows";
pub const DLL_PREFIX: &'static str = "";
pub const DLL_SUFFIX: &'static str = ".dll";
pub const DLL_EXTENSION: &'static str = "dll";
pub const EXE_SUFFIX: &'static str = ".exe";
pub const EXE_EXTENSION: &'static str = "exe";
