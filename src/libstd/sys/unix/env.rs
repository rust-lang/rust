// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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

use borrow::Cow;
use ffi::{CString, CStr, OsStr, OsString};
use sys::error::{self, Error, Result};
use sys::unix::{c, cvt};
use sys::env as sys;
use sys::fs;
use os::unix::ffi::{OsStrExt, OsStringExt};
use fmt;
use result;
use boxed::Box;
use string::String;
use vec::{self, Vec};
use io;
use iter;
use libc::{self, c_int, c_char, c_void};
use mem;
use ptr;
use slice;
use str;

pub use sys::common::env::JoinPathsError;
pub use sys::common::env::ARCH;

pub fn getcwd() -> Result<OsString> {
    let mut buf = Vec::with_capacity(512);
    loop {
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut libc::c_char;
            if !libc::getcwd(ptr, buf.capacity() as libc::size_t).is_null() {
                let len = CStr::from_ptr(buf.as_ptr() as *const libc::c_char).to_bytes().len();
                buf.set_len(len);
                buf.shrink_to_fit();
                return Ok(OsString::from_vec(buf));
            } else {
                let error = error::expect_last_error();
                if error.code() != libc::ERANGE {
                    return Err(error);
                }
            }

            // Trigger the internal buffer resizing logic of `Vec` by requiring
            // more space than the current capacity.
            let cap = buf.capacity();
            buf.set_len(cap);
            buf.reserve(1);
        }
    }
}

pub fn chdir(p: &OsStr) -> Result<()> {
    let p = try!(CString::new(p.as_bytes()));
    cvt(unsafe { libc::chdir(p.as_ptr()) }).map(drop)
}

pub fn getenv(k: &OsStr) -> Result<Option<OsString>> {
    unsafe {
        let s = try!(CString::new(k.as_bytes()));
        let s = libc::getenv(s.as_ptr()) as *const _;
        if s.is_null() {
            Ok(None)
        } else {
            Ok(Some(OsString::from_vec(CStr::from_ptr(s).to_bytes().to_vec())))
        }
    }
}

pub fn setenv(k: &OsStr, v: &OsStr) -> Result<()> {
    let k = try!(CString::new(k.as_bytes()));
    let v = try!(CString::new(v.as_bytes()));
    cvt(unsafe { libc::funcs::posix01::unistd::setenv(k.as_ptr(), v.as_ptr(), 1) }).map(drop)
}

pub fn unsetenv(k: &OsStr) -> Result<()> {
    let nbuf = try!(CString::new(k.as_bytes()));
    cvt(unsafe { libc::funcs::posix01::unistd::unsetenv(nbuf.as_ptr()) }).map(drop)
}

pub fn home_dir() -> Result<OsString> {
    return try!(getenv("HOME".as_ref())).map(Ok).unwrap_or_else(fallback);

    #[cfg(any(target_os = "android",
              target_os = "ios",
              target_os = "nacl"))]
    fn fallback() -> Result<OsString> { Err(Error::from_code(libc::ENOENT)) }
    #[cfg(not(any(target_os = "android",
                  target_os = "ios",
                  target_os = "nacl")))]
    fn fallback() -> Result<OsString> {
        unsafe {
            let amt = match libc::sysconf(c::_SC_GETPW_R_SIZE_MAX) {
                n if n < 0 => 512 as usize,
                n => n as usize,
            };
            let me = libc::getuid();
            loop {
                let mut buf = Vec::with_capacity(amt);
                let mut passwd: c::passwd = mem::zeroed();
                let mut result = ptr::null_mut();
                match c::getpwuid_r(me, &mut passwd, buf.as_mut_ptr(),
                                    buf.capacity() as libc::size_t,
                                    &mut result) {
                    0 if !result.is_null() => (),
                    0 => return Err(Error::from_code(libc::ENOENT)),
                    e => return Err(Error::from_code(e)),
                }
                let ptr = passwd.pw_dir as *const _;
                let bytes = CStr::from_ptr(ptr).to_bytes().to_vec();
                return Ok(OsString::from_vec(bytes))
            }
        }
    }
}

pub fn temp_dir() -> Result<OsString> {
    Ok(try!(getenv("TMPDIR".as_ref())).map(OsString::from).unwrap_or_else(|| {
        if cfg!(target_os = "android") {
            OsString::from_vec(String::from("/data/local/tmp").into())
        } else {
            OsString::from_vec(String::from("/tmp").into())
        }
    }))
}

pub fn vars() -> Result<Vars> {
    return unsafe {
        let mut environ = *environ();
        if environ as usize == 0 {
            return error::expect_last_result();
        }
        let mut result = Vec::new();
        while *environ != ptr::null() {
            result.push(parse(CStr::from_ptr(*environ).to_bytes()));
            environ = environ.offset(1);
        }
        Ok(Vars { iter: result.into_iter(), _dont_send_or_sync_me: ptr::null_mut() })
    };

    fn parse(input: &[u8]) -> (OsString, OsString) {
        let mut it = input.splitn(2, |b| *b == b'=');
        let key = it.next().unwrap().to_vec();
        let default: &[u8] = &[];
        let val = it.next().unwrap_or(default).to_vec();
        (OsString::from_vec(key), OsString::from_vec(val))
    }
}

/// Returns the command line arguments
///
/// Returns a list of the command line arguments.
#[cfg(target_os = "macos")]
pub fn args() -> Result<Args> {
    extern {
        // These functions are in crt_externs.h.
        fn _NSGetArgc() -> *mut c_int;
        fn _NSGetArgv() -> *mut *mut *mut c_char;
    }

    let vec = unsafe {
        let (argc, argv) = (*_NSGetArgc() as isize,
                            *_NSGetArgv() as *const *const c_char);
        (0.. argc as isize).map(|i| {
            let bytes = CStr::from_ptr(*argv.offset(i)).to_bytes().to_vec();
            OsString::from_vec(bytes)
        }).collect::<Vec<_>>()
    };
    Ok(Args {
        iter: vec.into_iter(),
        _dont_send_or_sync_me: ptr::null_mut(),
    })
}

// As _NSGetArgc and _NSGetArgv aren't mentioned in iOS docs
// and use underscores in their names - they're most probably
// are considered private and therefore should be avoided
// Here is another way to get arguments using Objective C
// runtime
//
// In general it looks like:
// res = Vec::new()
// let args = [[NSProcessInfo processInfo] arguments]
// for i in (0..[args count])
//      res.push([args objectAtIndex:i])
// res
#[cfg(target_os = "ios")]
pub fn args() -> Result<Args> {
    use mem;

    #[link(name = "objc")]
    extern {
        fn sel_registerName(name: *const libc::c_uchar) -> Sel;
        fn objc_msgSend(obj: NsId, sel: Sel, ...) -> NsId;
        fn objc_getClass(class_name: *const libc::c_uchar) -> NsId;
    }

    #[link(name = "Foundation", kind = "framework")]
    extern {}

    type Sel = *const libc::c_void;
    type NsId = *const libc::c_void;

    let mut res = Vec::new();

    unsafe {
        let process_info_sel = sel_registerName("processInfo\0".as_ptr());
        let arguments_sel = sel_registerName("arguments\0".as_ptr());
        let utf8_sel = sel_registerName("UTF8String\0".as_ptr());
        let count_sel = sel_registerName("count\0".as_ptr());
        let object_at_sel = sel_registerName("objectAtIndex:\0".as_ptr());

        let klass = objc_getClass("NSProcessInfo\0".as_ptr());
        let info = objc_msgSend(klass, process_info_sel);
        let args = objc_msgSend(info, arguments_sel);

        let cnt: usize = mem::transmute(objc_msgSend(args, count_sel));
        for i in 0..cnt {
            let tmp = objc_msgSend(args, object_at_sel, i);
            let utf_c_str: *const libc::c_char =
                mem::transmute(objc_msgSend(tmp, utf8_sel));
            let bytes = CStr::from_ptr(utf_c_str).to_bytes();
            res.push(OsString::from(str::from_utf8(bytes).unwrap()))
        }
    }

    Ok(Args { iter: res.into_iter(), _dont_send_or_sync_me: ptr::null_mut() })
}

#[cfg(any(target_os = "linux",
          target_os = "android",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
          target_os = "openbsd",
          target_os = "nacl"))]
pub fn args() -> Result<Args> {
    use sys::unix::args;
    let bytes = args::clone().unwrap_or(Vec::new());
    let v: Vec<OsString> = bytes.into_iter().map(|v| {
        OsString::from_vec(v)
    }).collect();
    Ok(Args { iter: v.into_iter(), _dont_send_or_sync_me: ptr::null_mut() })
}

pub fn join_paths<I: Iterator<Item=T>, T: AsRef<OsStr>>(paths: I) -> result::Result<OsString, sys::JoinPathsError> {
    let mut joined = Vec::new();
    let sep = b':';

    for (i, path) in paths.enumerate() {
        let path = path.as_ref().as_bytes();
        if i > 0 { joined.push(sep) }
        if path.contains(&sep) {
            return Err(sys::JoinPathsError::new())
        }
        joined.push_all(path);
    }
    Ok(OsString::from_vec(joined.into()))
}

pub fn join_paths_error() -> &'static str { "path segment contains separator `:`" }

pub const FAMILY: &'static str = os::FAMILY;
pub const OS: &'static str = os::OS;
pub const DLL_PREFIX: &'static str = os::DLL_PREFIX;
pub const DLL_SUFFIX: &'static str = os::DLL_SUFFIX;
pub const DLL_EXTENSION: &'static str = os::DLL_EXTENSION;
pub const EXE_SUFFIX: &'static str = os::EXE_SUFFIX;
pub const EXE_EXTENSION: &'static str = os::EXE_EXTENSION;

pub fn split_paths(unparsed: &OsStr) -> SplitPaths {
    fn bytes_to_path(b: &[u8]) -> &OsStr {
        OsStr::from_bytes(b)
    }
    fn is_colon(b: &u8) -> bool { *b == b':' }
    let unparsed = unparsed.as_bytes();
    SplitPaths {
        iter: unparsed.split(is_colon as fn(&u8) -> bool)
                      .map(bytes_to_path as fn(&[u8]) -> &OsStr)
    }
}

pub struct SplitPaths<'a> {
    iter: iter::Map<slice::Split<'a, u8, fn(&u8) -> bool>,
                    fn(&'a [u8]) -> &'a OsStr>,
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = Cow<'a, OsStr>;

    fn next(&mut self) -> Option<Cow<'a, OsStr>> { self.iter.next().map(Cow::Borrowed) }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

pub struct Args {
    iter: vec::IntoIter<OsString>,
    _dont_send_or_sync_me: *mut (),
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize { self.iter.len() }
}


pub struct Vars {
    iter: vec::IntoIter<(OsString, OsString)>,
    _dont_send_or_sync_me: *mut (),
}

impl Iterator for Vars {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

#[cfg(target_os = "freebsd")]
pub fn current_exe() -> Result<OsString> {
    unsafe {
        use libc::funcs::bsd44::*;
        use libc::consts::os::extra::*;
        let mut mib = [CTL_KERN as c_int,
                       KERN_PROC as c_int,
                       KERN_PROC_PATHNAME as c_int,
                       -1 as c_int];
        let mut sz: libc::size_t = 0;
        let err = sysctl(mib.as_mut_ptr(), mib.len() as ::libc::c_uint,
                         ptr::null_mut(), &mut sz, ptr::null_mut(),
                         0 as libc::size_t);
        if err != 0 { return error::expect_last_result(); }
        if sz == 0 { return error::expect_last_result(); }
        let mut v: Vec<u8> = Vec::with_capacity(sz as usize);
        let err = sysctl(mib.as_mut_ptr(), mib.len() as ::libc::c_uint,
                         v.as_mut_ptr() as *mut libc::c_void, &mut sz,
                         ptr::null_mut(), 0 as libc::size_t);
        if err != 0 { return error::expect_last_error(); }
        if sz == 0 { return error::expect_last_error(); }
        v.set_len(sz as usize - 1); // chop off trailing NUL
        Ok(OsString::from(OsString::from_vec(v)))
    }
}

#[cfg(target_os = "dragonfly")]
pub fn current_exe() -> Result<OsString> {
    fs::readlink(OsStr::from_str("/proc/curproc/file"))
}

#[cfg(target_os = "netbsd")]
pub fn current_exe() -> Result<OsString> {
    fs::readlink(OsStr::from_str("/proc/curproc/exe"))
}

#[cfg(any(target_os = "bitrig", target_os = "openbsd"))]
pub fn current_exe() -> Result<OsString> {
    use sync::StaticMutex;
    static LOCK: StaticMutex = StaticMutex::new();

    extern {
        fn rust_current_exe() -> *const c_char;
    }

    let _guard = LOCK.lock();

    unsafe {
        let v = rust_current_exe();
        if v.is_null() {
            error::expect_last_result()
        } else {
            let vec = CStr::from_ptr(v).to_bytes().to_vec();
            Ok(OsString::from(OsString::from_vec(vec)))
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn current_exe() -> Result<OsString> {
    fs::readlink("/proc/self/exe".as_ref())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn current_exe() -> Result<OsString> {
    unsafe {
        use libc::funcs::extra::_NSGetExecutablePath;
        let mut sz: u32 = 0;
        _NSGetExecutablePath(ptr::null_mut(), &mut sz);
        if sz == 0 { return error::expect_last_result(); }
        let mut v: Vec<u8> = Vec::with_capacity(sz as usize);
        let err = _NSGetExecutablePath(v.as_mut_ptr() as *mut i8, &mut sz);
        if err != 0 { return error::expect_last_result(); }
        v.set_len(sz as usize - 1); // chop off trailing NUL
        Ok(OsString::from_vec(v))
    }
}

#[cfg(target_os = "macos")]
pub unsafe fn environ() -> *mut *const *const c_char {
    extern { fn _NSGetEnviron() -> *mut *const *const c_char; }
    _NSGetEnviron()
}

#[cfg(not(target_os = "macos"))]
pub unsafe fn environ() -> *mut *const *const c_char {
    extern { static mut environ: *const *const c_char; }
    &mut environ
}

#[cfg(target_os = "linux")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "linux";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "macos")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "macos";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".dylib";
    pub const DLL_EXTENSION: &'static str = "dylib";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "ios")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "ios";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".dylib";
    pub const DLL_EXTENSION: &'static str = "dylib";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "freebsd")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "freebsd";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "dragonfly")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "dragonfly";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "bitrig")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "bitrig";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "netbsd")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "netbsd";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "openbsd")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "openbsd";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(target_os = "android")]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "android";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = "";
    pub const EXE_EXTENSION: &'static str = "";
}

#[cfg(all(target_os = "nacl", not(target_arch = "le32")))]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "nacl";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".so";
    pub const DLL_EXTENSION: &'static str = "so";
    pub const EXE_SUFFIX: &'static str = ".nexe";
    pub const EXE_EXTENSION: &'static str = "nexe";
}

#[cfg(all(target_os = "nacl", target_arch = "le32"))]
mod os {
    pub const FAMILY: &'static str = "unix";
    pub const OS: &'static str = "pnacl";
    pub const DLL_PREFIX: &'static str = "lib";
    pub const DLL_SUFFIX: &'static str = ".pso";
    pub const DLL_EXTENSION: &'static str = "pso";
    pub const EXE_SUFFIX: &'static str = ".pexe";
    pub const EXE_EXTENSION: &'static str = "pexe";
}
