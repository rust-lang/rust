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

use prelude::v1::*;
use os::windows::prelude::*;

use error::Error as StdError;
use ffi::{OsString, OsStr};
use fmt;
use io;
use libc::{c_int, c_void};
use ops::Range;
use os::windows::ffi::EncodeWide;
use path::{self, PathBuf};
use ptr;
use slice;
use sys::{c, cvt};
use sys::handle::Handle;

use super::to_u16s;

pub fn errno() -> i32 {
    unsafe { c::GetLastError() as i32 }
}

/// Gets a detailed string description for the given error number.
pub fn error_string(errnum: i32) -> String {
    // This value is calculated from the macro
    // MAKELANGID(LANG_SYSTEM_DEFAULT, SUBLANG_SYS_DEFAULT)
    let langId = 0x0800 as c::DWORD;

    let mut buf = [0 as c::WCHAR; 2048];

    unsafe {
        let res = c::FormatMessageW(c::FORMAT_MESSAGE_FROM_SYSTEM |
                                        c::FORMAT_MESSAGE_IGNORE_INSERTS,
                                    ptr::null_mut(),
                                    errnum as c::DWORD,
                                    langId,
                                    buf.as_mut_ptr(),
                                    buf.len() as c::DWORD,
                                    ptr::null()) as usize;
        if res == 0 {
            // Sometimes FormatMessageW can fail e.g. system doesn't like langId,
            let fm_err = errno();
            return format!("OS Error {} (FormatMessageW() returned error {})",
                           errnum, fm_err);
        }

        match String::from_utf16(&buf[..res]) {
            Ok(mut msg) => {
                // Trim trailing CRLF inserted by FormatMessageW
                let len = msg.trim_right().len();
                msg.truncate(len);
                msg
            },
            Err(..) => format!("OS Error {} (FormatMessageW() returned \
                                invalid UTF-16)", errnum),
        }
    }
}

pub struct Env {
    base: c::LPWCH,
    cur: c::LPWCH,
}

impl Iterator for Env {
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

impl Drop for Env {
    fn drop(&mut self) {
        unsafe { c::FreeEnvironmentStringsW(self.base); }
    }
}

pub fn env() -> Env {
    unsafe {
        let ch = c::GetEnvironmentStringsW();
        if ch as usize == 0 {
            panic!("failure getting env string from OS: {}",
                   io::Error::last_os_error());
        }
        Env { base: ch, cur: ch }
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
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
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
            Some(super::os2path(&in_progress))
        }
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
    where I: Iterator<Item=T>, T: AsRef<OsStr>
{
    let mut joined = Vec::new();
    let sep = b';' as u16;

    for (i, path) in paths.enumerate() {
        let path = path.as_ref();
        if i > 0 { joined.push(sep) }
        let v = path.encode_wide().collect::<Vec<u16>>();
        if v.contains(&(b'"' as u16)) {
            return Err(JoinPathsError)
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

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "path segment contains `\"`".fmt(f)
    }
}

impl StdError for JoinPathsError {
    fn description(&self) -> &str { "failed to join paths" }
}

pub fn current_exe() -> io::Result<PathBuf> {
    super::fill_utf16_buf(|buf, sz| unsafe {
        c::GetModuleFileNameW(ptr::null_mut(), buf, sz)
    }, super::os2path)
}

pub fn getcwd() -> io::Result<PathBuf> {
    super::fill_utf16_buf(|buf, sz| unsafe {
        c::GetCurrentDirectoryW(sz, buf)
    }, super::os2path)
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    let p: &OsStr = p.as_ref();
    let mut p = p.encode_wide().collect::<Vec<_>>();
    p.push(0);

    cvt(unsafe {
        c::SetCurrentDirectoryW(p.as_ptr())
    }).map(|_| ())
}

pub fn getenv(k: &OsStr) -> io::Result<Option<OsString>> {
    let k = try!(to_u16s(k));
    let res = super::fill_utf16_buf(|buf, sz| unsafe {
        c::GetEnvironmentVariableW(k.as_ptr(), buf, sz)
    }, |buf| {
        OsStringExt::from_wide(buf)
    });
    match res {
        Ok(value) => Ok(Some(value)),
        Err(e) => {
            if e.raw_os_error() == Some(c::ERROR_ENVVAR_NOT_FOUND as i32) {
                Ok(None)
            } else {
                Err(e)
            }
        }
    }
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let k = try!(to_u16s(k));
    let v = try!(to_u16s(v));

    cvt(unsafe {
        c::SetEnvironmentVariableW(k.as_ptr(), v.as_ptr())
    }).map(|_| ())
}

pub fn unsetenv(n: &OsStr) -> io::Result<()> {
    let v = try!(to_u16s(n));
    cvt(unsafe {
        c::SetEnvironmentVariableW(v.as_ptr(), ptr::null())
    }).map(|_| ())
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

pub fn args() -> Args {
    unsafe {
        let mut nArgs: c_int = 0;
        let lpCmdLine = c::GetCommandLineW();
        let szArgList = c::CommandLineToArgvW(lpCmdLine, &mut nArgs);

        // szArcList can be NULL if CommandLinToArgvW failed,
        // but in that case nArgs is 0 so we won't actually
        // try to read a null pointer
        Args { cur: szArgList, range: 0..(nArgs as isize) }
    }
}

pub fn temp_dir() -> PathBuf {
    super::fill_utf16_buf(|buf, sz| unsafe {
        c::GetTempPathW(sz, buf)
    }, super::os2path).unwrap()
}

pub fn home_dir() -> Option<PathBuf> {
    ::env::var_os("HOME").or_else(|| {
        ::env::var_os("USERPROFILE")
    }).map(PathBuf::from).or_else(|| unsafe {
        let me = c::GetCurrentProcess();
        let mut token = ptr::null_mut();
        if c::OpenProcessToken(me, c::TOKEN_READ, &mut token) == 0 {
            return None
        }
        let _handle = Handle::new(token);
        super::fill_utf16_buf(|buf, mut sz| {
            match c::GetUserProfileDirectoryW(token, buf, &mut sz) {
                0 if c::GetLastError() != 0 => 0,
                0 => sz,
                n => n as c::DWORD,
            }
        }, super::os2path).ok()
    })
}

pub fn exit(code: i32) -> ! {
    unsafe { c::ExitProcess(code as c::UINT) }
}
