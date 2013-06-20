// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Cross-platform file path handling

*/

#[allow(missing_doc)];

use container::Container;
use cmp::Eq;
use iterator::IteratorUtil;
use libc;
use option::{None, Option, Some};
use str;
use str::{Str, StrSlice, StrVector};
use to_str::ToStr;
use ascii::{AsciiCast, AsciiStr};
use old_iter::BaseIter;
use vec::OwnedVector;

#[cfg(windows)]
pub use Path = self::WindowsPath;
#[cfg(unix)]
pub use Path = self::PosixPath;

#[deriving(Clone, Eq)]
pub struct WindowsPath {
    host: Option<~str>,
    device: Option<~str>,
    is_absolute: bool,
    components: ~[~str],
}

pub fn WindowsPath(s: &str) -> WindowsPath {
    GenericPath::from_str(s)
}

#[deriving(Clone, Eq)]
pub struct PosixPath {
    is_absolute: bool,
    components: ~[~str],
}

pub fn PosixPath(s: &str) -> PosixPath {
    GenericPath::from_str(s)
}

pub trait GenericPath {
    /// Converts a string to a Path
    fn from_str(&str) -> Self;

    /// Returns the directory component of `self`, as a string
    fn dirname(&self) -> ~str;
    /// Returns the file component of `self`, as a string option.
    /// Returns None if `self` names a directory.
    fn filename(&self) -> Option<~str>;
    /// Returns the stem of the file component of `self`, as a string option.
    /// The stem is the slice of a filename starting at 0 and ending just before
    /// the last '.' in the name.
    /// Returns None if `self` names a directory.
    fn filestem(&self) -> Option<~str>;
    /// Returns the type of the file component of `self`, as a string option.
    /// The file type is the slice of a filename starting just after the last
    /// '.' in the name and ending at the last index in the filename.
    /// Returns None if `self` names a directory.
    fn filetype(&self) -> Option<~str>;

    /// Returns a new path consisting of `self` with the parent directory component replaced
    /// with the given string.
    fn with_dirname(&self, (&str)) -> Self;
    /// Returns a new path consisting of `self` with the file component replaced
    /// with the given string.
    fn with_filename(&self, (&str)) -> Self;
    /// Returns a new path consisting of `self` with the file stem replaced
    /// with the given string.
    fn with_filestem(&self, (&str)) -> Self;
    /// Returns a new path consisting of `self` with the file type replaced
    /// with the given string.
    fn with_filetype(&self, (&str)) -> Self;

    /// Returns the directory component of `self`, as a new path.
    /// If `self` has no parent, returns `self`.
    fn dir_path(&self) -> Self;
    /// Returns the file component of `self`, as a new path.
    /// If `self` names a directory, returns the empty path.
    fn file_path(&self) -> Self;

    /// Returns a new Path whose parent directory is `self` and whose
    /// file component is the given string.
    fn push(&self, (&str)) -> Self;
    /// Returns a new Path consisting of the given path, made relative to `self`.
    fn push_rel(&self, (&Self)) -> Self;
    /// Returns a new Path consisting of the path given by the given vector
    /// of strings, relative to `self`.
    fn push_many<S: Str>(&self, (&[S])) -> Self;
    /// Identical to `dir_path` except in the case where `self` has only one
    /// component. In this case, `pop` returns the empty path.
    fn pop(&self) -> Self;

    /// The same as `push_rel`, except that the directory argument must not
    /// contain directory separators in any of its components.
    fn unsafe_join(&self, (&Self)) -> Self;
    /// On Unix, always returns false. On Windows, returns true iff `self`'s
    /// file stem is one of: `con` `aux` `com1` `com2` `com3` `com4`
    /// `lpt1` `lpt2` `lpt3` `prn` `nul`
    fn is_restricted(&self) -> bool;

    /// Returns a new path that names the same file as `self`, without containing
    /// any '.', '..', or empty components. On Windows, uppercases the drive letter
    /// as well.
    fn normalize(&self) -> Self;

    /// Returns `true` if `self` is an absolute path.
    fn is_absolute(&self) -> bool;
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
mod stat {
    #[cfg(target_arch = "x86")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                __pad1: 0,
                st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                __pad2: 0,
                st_size: 0,
                st_blksize: 0,
                st_blocks: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                __unused4: 0,
                __unused5: 0,
            }
        }
    }

    #[cfg(target_arch = "arm")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                __pad0: [0, ..4],
                __st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                __pad3: [0, ..4],
                st_size: 0,
                st_blksize: 0,
                st_blocks: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                st_ino: 0
            }
        }
    }

    #[cfg(target_arch = "mips")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_pad1: [0, ..3],
                st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                st_pad2: [0, ..2],
                st_size: 0,
                st_pad3: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                st_blksize: 0,
                st_blocks: 0,
                st_pad5: [0, ..14],
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_ino: 0,
                st_nlink: 0,
                st_mode: 0,
                st_uid: 0,
                st_gid: 0,
                __pad0: 0,
                st_rdev: 0,
                st_size: 0,
                st_blksize: 0,
                st_blocks: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                __unused: [0, 0, 0],
            }
        }
    }
}

#[cfg(target_os = "freebsd")]
mod stat {
    #[cfg(target_arch = "x86_64")]
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                st_size: 0,
                st_blocks: 0,
                st_blksize: 0,
                st_flags: 0,
                st_gen: 0,
                st_lspare: 0,
                st_birthtime: 0,
                st_birthtime_nsec: 0,
                __unused: [0, 0],
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod stat {
    pub mod arch {
        use libc;

        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_mode: 0,
                st_nlink: 0,
                st_ino: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                st_atime: 0,
                st_atime_nsec: 0,
                st_mtime: 0,
                st_mtime_nsec: 0,
                st_ctime: 0,
                st_ctime_nsec: 0,
                st_birthtime: 0,
                st_birthtime_nsec: 0,
                st_size: 0,
                st_blocks: 0,
                st_blksize: 0,
                st_flags: 0,
                st_gen: 0,
                st_lspare: 0,
                st_qspare: [0, 0],
            }
        }
    }
}

#[cfg(target_os = "win32")]
mod stat {
    pub mod arch {
        use libc;
        pub fn default_stat() -> libc::stat {
            libc::stat {
                st_dev: 0,
                st_ino: 0,
                st_mode: 0,
                st_nlink: 0,
                st_uid: 0,
                st_gid: 0,
                st_rdev: 0,
                st_size: 0,
                st_atime: 0,
                st_mtime: 0,
                st_ctime: 0,
            }
        }
    }
}


impl Path {
    pub fn stat(&self) -> Option<libc::stat> {
        unsafe {
             do str::as_c_str(self.to_str()) |buf| {
                let mut st = stat::arch::default_stat();
                match libc::stat(buf, &mut st) {
                    0 => Some(st),
                    _ => None,
                }
            }
        }
    }

    #[cfg(unix)]
    pub fn lstat(&self) -> Option<libc::stat> {
        unsafe {
            do str::as_c_str(self.to_str()) |buf| {
                let mut st = stat::arch::default_stat();
                match libc::lstat(buf, &mut st) {
                    0 => Some(st),
                    _ => None,
                }
            }
        }
    }

    pub fn exists(&self) -> bool {
        match self.stat() {
            None => false,
            Some(_) => true,
        }
    }

    pub fn get_size(&self) -> Option<i64> {
        match self.stat() {
            None => None,
            Some(ref st) => Some(st.st_size as i64),
        }
    }

    pub fn get_mode(&self) -> Option<uint> {
        match self.stat() {
            None => None,
            Some(ref st) => Some(st.st_mode as uint),
        }
    }
}

#[cfg(target_os = "freebsd")]
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
impl Path {
    pub fn get_atime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_atime as i64,
                      st.st_atime_nsec as int))
            }
        }
    }

    pub fn get_mtime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_mtime as i64,
                      st.st_mtime_nsec as int))
            }
        }
    }

    pub fn get_ctime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_ctime as i64,
                      st.st_ctime_nsec as int))
            }
        }
    }
}

#[cfg(target_os = "freebsd")]
#[cfg(target_os = "macos")]
impl Path {
    pub fn get_birthtime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_birthtime as i64,
                      st.st_birthtime_nsec as int))
            }
        }
    }
}

#[cfg(target_os = "win32")]
impl Path {
    pub fn get_atime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_atime as i64, 0))
            }
        }
    }

    pub fn get_mtime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_mtime as i64, 0))
            }
        }
    }

    pub fn get_ctime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_ctime as i64, 0))
            }
        }
    }
}

impl ToStr for PosixPath {
    fn to_str(&self) -> ~str {
        let mut s = ~"";
        if self.is_absolute {
            s += "/";
        }
        s + self.components.connect("/")
    }
}

// FIXME (#3227): when default methods in traits are working, de-duplicate
// PosixPath and WindowsPath, most of their methods are common.
impl GenericPath for PosixPath {
    fn from_str(s: &str) -> PosixPath {
        let components = s.split_iter('/')
            .filter_map(|s| if s.is_empty() {None} else {Some(s.to_owned())})
            .collect();
        let is_absolute = (s.len() != 0 && s[0] == '/' as u8);
        PosixPath {
            is_absolute: is_absolute,
            components: components,
        }
    }

    fn dirname(&self) -> ~str {
        let s = self.dir_path().to_str();
        match s.len() {
            0 => ~".",
            _ => s,
        }
    }

    fn filename(&self) -> Option<~str> {
        match self.components.len() {
            0 => None,
            n => Some(copy self.components[n - 1]),
        }
    }

    fn filestem(&self) -> Option<~str> {
        match self.filename() {
            None => None,
            Some(ref f) => {
                match f.rfind('.') {
                    Some(p) => Some(f.slice_to(p).to_owned()),
                    None => Some(copy *f),
                }
            }
        }
    }

    fn filetype(&self) -> Option<~str> {
        match self.filename() {
            None => None,
            Some(ref f) => {
                match f.rfind('.') {
                    Some(p) if p < f.len() => Some(f.slice_from(p).to_owned()),
                    _ => None,
                }
            }
        }
    }

    fn with_dirname(&self, d: &str) -> PosixPath {
        let dpath = PosixPath(d);
        match self.filename() {
            Some(ref f) => dpath.push(*f),
            None => dpath,
        }
    }

    fn with_filename(&self, f: &str) -> PosixPath {
        assert!(! f.iter().all(windows::is_sep));
        self.dir_path().push(f)
    }

    fn with_filestem(&self, s: &str) -> PosixPath {
        match self.filetype() {
            None => self.with_filename(s),
            Some(ref t) => self.with_filename(s.to_owned() + *t),
        }
    }

    fn with_filetype(&self, t: &str) -> PosixPath {
        match (t.len(), self.filestem()) {
            (0, None)        => copy *self,
            (0, Some(ref s)) => self.with_filename(*s),
            (_, None)        => self.with_filename(fmt!(".%s", t)),
            (_, Some(ref s)) => self.with_filename(fmt!("%s.%s", *s, t)),
        }
    }

    fn dir_path(&self) -> PosixPath {
        match self.components.len() {
            0 => copy *self,
            _ => self.pop(),
        }
    }

    fn file_path(&self) -> PosixPath {
        let cs = match self.filename() {
          None => ~[],
          Some(ref f) => ~[copy *f]
        };
        PosixPath {
            is_absolute: false,
            components: cs,
        }
    }

    fn push_rel(&self, other: &PosixPath) -> PosixPath {
        assert!(!other.is_absolute);
        self.push_many(other.components)
    }

    fn unsafe_join(&self, other: &PosixPath) -> PosixPath {
        if other.is_absolute {
            PosixPath {
                is_absolute: true,
                components: copy other.components,
            }
        } else {
            self.push_rel(other)
        }
    }

    fn is_restricted(&self) -> bool {
        false
    }

    fn push_many<S: Str>(&self, cs: &[S]) -> PosixPath {
        let mut v = copy self.components;
        for cs.each |e| {
            for e.as_slice().split_iter(windows::is_sep).advance |s| {
                if !s.is_empty() {
                    v.push(s.to_owned())
                }
            }
        }
        PosixPath {
            is_absolute: self.is_absolute,
            components: v,
        }
    }

    fn push(&self, s: &str) -> PosixPath {
        let mut v = copy self.components;
        for s.split_iter(windows::is_sep).advance |s| {
            if !s.is_empty() {
                v.push(s.to_owned())
            }
        }
        PosixPath { components: v, ..copy *self }
    }

    fn pop(&self) -> PosixPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            cs.pop();
        }
        PosixPath {
            is_absolute: self.is_absolute,
            components: cs,
        } //..self }
    }

    fn normalize(&self) -> PosixPath {
        PosixPath {
            is_absolute: self.is_absolute,
            components: normalize(self.components),
        } // ..self }
    }

    fn is_absolute(&self) -> bool {
        self.is_absolute
    }
}


impl ToStr for WindowsPath {
    fn to_str(&self) -> ~str {
        let mut s = ~"";
        match self.host {
          Some(ref h) => { s += "\\\\"; s += *h; }
          None => { }
        }
        match self.device {
          Some(ref d) => { s += *d; s += ":"; }
          None => { }
        }
        if self.is_absolute {
            s += "\\";
        }
        s + self.components.connect("\\")
    }
}


impl GenericPath for WindowsPath {
    fn from_str(s: &str) -> WindowsPath {
        let host;
        let device;
        let rest;

        match (
            windows::extract_drive_prefix(s),
            windows::extract_unc_prefix(s),
        ) {
            (Some((ref d, ref r)), _) => {
                host = None;
                device = Some(copy *d);
                rest = copy *r;
            }
            (None, Some((ref h, ref r))) => {
                host = Some(copy *h);
                device = None;
                rest = copy *r;
            }
            (None, None) => {
                host = None;
                device = None;
                rest = s.to_owned();
            }
        }

        let components = rest.split_iter(windows::is_sep)
            .filter_map(|s| if s.is_empty() {None} else {Some(s.to_owned())})
            .collect();

        let is_absolute = (rest.len() != 0 && windows::is_sep(rest[0] as char));
        WindowsPath {
            host: host,
            device: device,
            is_absolute: is_absolute,
            components: components,
        }
    }

    fn dirname(&self) -> ~str {
        let s = self.dir_path().to_str();
        match s.len() {
            0 => ~".",
            _ => s,
        }
    }

    fn filename(&self) -> Option<~str> {
        match self.components.len() {
            0 => None,
            n => Some(copy self.components[n - 1]),
        }
    }

    fn filestem(&self) -> Option<~str> {
        match self.filename() {
            None => None,
            Some(ref f) => {
                match f.rfind('.') {
                    Some(p) => Some(f.slice_to(p).to_owned()),
                    None => Some(copy *f),
                }
            }
        }
    }

    fn filetype(&self) -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match f.rfind('.') {
                Some(p) if p < f.len() => Some(f.slice_from(p).to_owned()),
                _ => None,
            }
          }
        }
    }

    fn with_dirname(&self, d: &str) -> WindowsPath {
        let dpath = WindowsPath(d);
        match self.filename() {
            Some(ref f) => dpath.push(*f),
            None => dpath,
        }
    }

    fn with_filename(&self, f: &str) -> WindowsPath {
        assert!(! f.iter().all(windows::is_sep));
        self.dir_path().push(f)
    }

    fn with_filestem(&self, s: &str) -> WindowsPath {
        match self.filetype() {
            None => self.with_filename(s),
            Some(ref t) => self.with_filename(s.to_owned() + *t),
        }
    }

    fn with_filetype(&self, t: &str) -> WindowsPath {
        match (t.len(), self.filestem()) {
            (0, None)        => copy *self,
            (0, Some(ref s)) => self.with_filename(*s),
            (_, None)        => self.with_filename(fmt!(".%s", t)),
            (_, Some(ref s)) => self.with_filename(fmt!("%s.%s", *s, t)),
        }
    }

    fn dir_path(&self) -> WindowsPath {
        match self.components.len() {
            0 => copy *self,
            _ => self.pop(),
        }
    }

    fn file_path(&self) -> WindowsPath {
        WindowsPath {
            host: None,
            device: None,
            is_absolute: false,
            components: match self.filename() {
                None => ~[],
                Some(ref f) => ~[copy *f],
            }
        }
    }

    fn push_rel(&self, other: &WindowsPath) -> WindowsPath {
        assert!(!other.is_absolute);
        self.push_many(other.components)
    }

    fn unsafe_join(&self, other: &WindowsPath) -> WindowsPath {
        /* rhs not absolute is simple push */
        if !other.is_absolute {
            return self.push_many(other.components);
        }

        /* if rhs has a host set, then the whole thing wins */
        match other.host {
            Some(ref host) => {
                return WindowsPath {
                    host: Some(copy *host),
                    device: copy other.device,
                    is_absolute: true,
                    components: copy other.components,
                };
            }
            _ => {}
        }

        /* if rhs has a device set, then a part wins */
        match other.device {
            Some(ref device) => {
                return WindowsPath {
                    host: None,
                    device: Some(copy *device),
                    is_absolute: true,
                    components: copy other.components,
                };
            }
            _ => {}
        }

        /* fallback: host and device of lhs win, but the
           whole path of the right */
        WindowsPath {
            host: copy self.host,
            device: copy self.device,
            is_absolute: self.is_absolute || other.is_absolute,
            components: copy other.components,
        }
    }

    fn is_restricted(&self) -> bool {
        match self.filestem() {
            Some(stem) => {
                // FIXME: #4318 Instead of to_ascii and to_str_ascii, could use
                // to_ascii_consume and to_str_consume to not do a unnecessary copy.
                match stem.to_ascii().to_lower().to_str_ascii() {
                    ~"con" | ~"aux" | ~"com1" | ~"com2" | ~"com3" | ~"com4" |
                    ~"lpt1" | ~"lpt2" | ~"lpt3" | ~"prn" | ~"nul" => true,
                    _ => false
                }
            },
            None => false
        }
    }

    fn push_many<S: Str>(&self, cs: &[S]) -> WindowsPath {
        let mut v = copy self.components;
        for cs.each |e| {
            for e.as_slice().split_iter(windows::is_sep).advance |s| {
                if !s.is_empty() {
                    v.push(s.to_owned())
                }
            }
        }
        // tedious, but as-is, we can't use ..self
        WindowsPath {
            host: copy self.host,
            device: copy self.device,
            is_absolute: self.is_absolute,
            components: v
        }
    }

    fn push(&self, s: &str) -> WindowsPath {
        let mut v = copy self.components;
        for s.split_iter(windows::is_sep).advance |s| {
            if !s.is_empty() {
                v.push(s.to_owned())
            }
        }
        WindowsPath { components: v, ..copy *self }
    }

    fn pop(&self) -> WindowsPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            cs.pop();
        }
        WindowsPath {
            host: copy self.host,
            device: copy self.device,
            is_absolute: self.is_absolute,
            components: cs,
        }
    }

    fn normalize(&self) -> WindowsPath {
        WindowsPath {
            host: copy self.host,
            device: match self.device {
                None => None,

                // FIXME: #4318 Instead of to_ascii and to_str_ascii, could use
                // to_ascii_consume and to_str_consume to not do a unnecessary copy.
                Some(ref device) => Some(device.to_ascii().to_upper().to_str_ascii())
            },
            is_absolute: self.is_absolute,
            components: normalize(self.components)
        }
    }

    fn is_absolute(&self) -> bool {
        self.is_absolute
    }
}


pub fn normalize(components: &[~str]) -> ~[~str] {
    let mut cs = ~[];
    for components.each |c| {
        if *c == ~"." && components.len() > 1 { loop; }
        if *c == ~"" { loop; }
        if *c == ~".." && cs.len() != 0 {
            cs.pop();
            loop;
        }
        cs.push(copy *c);
    }
    cs
}

// Various windows helpers, and tests for the impl.
pub mod windows {
    use libc;
    use option::{None, Option, Some};

    #[inline]
    pub fn is_sep(u: char) -> bool {
        u == '/' || u == '\\'
    }

    pub fn extract_unc_prefix(s: &str) -> Option<(~str,~str)> {
        if (s.len() > 1 &&
            (s[0] == '\\' as u8 || s[0] == '/' as u8) &&
            s[0] == s[1]) {
            let mut i = 2;
            while i < s.len() {
                if is_sep(s[i] as char) {
                    let pre = s.slice(2, i).to_owned();
                    let rest = s.slice(i, s.len()).to_owned();
                    return Some((pre, rest));
                }
                i += 1;
            }
        }
        None
    }

    pub fn extract_drive_prefix(s: &str) -> Option<(~str,~str)> {
        unsafe {
            if (s.len() > 1 &&
                libc::isalpha(s[0] as libc::c_int) != 0 &&
                s[1] == ':' as u8) {
                let rest = if s.len() == 2 {
                    ~""
                } else {
                    s.slice(2, s.len()).to_owned()
                };
                return Some((s.slice(0,1).to_owned(), rest));
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use option::{None, Some};
    use path::{PosixPath, WindowsPath, windows};

    #[test]
    fn test_double_slash_collapsing() {
        let path = PosixPath("tmp/");
        let path = path.push("/hmm");
        let path = path.normalize();
        assert_eq!(~"tmp/hmm", path.to_str());

        let path = WindowsPath("tmp/");
        let path = path.push("/hmm");
        let path = path.normalize();
        assert_eq!(~"tmp\\hmm", path.to_str());
    }

    #[test]
    fn test_filetype_foo_bar() {
        let wp = PosixPath("foo.bar");
        assert_eq!(wp.filetype(), Some(~".bar"));

        let wp = WindowsPath("foo.bar");
        assert_eq!(wp.filetype(), Some(~".bar"));
    }

    #[test]
    fn test_filetype_foo() {
        let wp = PosixPath("foo");
        assert_eq!(wp.filetype(), None);

        let wp = WindowsPath("foo");
        assert_eq!(wp.filetype(), None);
    }

    #[test]
    fn test_posix_paths() {
        fn t(wp: &PosixPath, s: &str) {
            let ss = wp.to_str();
            let sss = s.to_owned();
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert_eq!(ss, sss);
            }
        }

        t(&(PosixPath("hi")), "hi");
        t(&(PosixPath("/lib")), "/lib");
        t(&(PosixPath("hi/there")), "hi/there");
        t(&(PosixPath("hi/there.txt")), "hi/there.txt");

        t(&(PosixPath("hi/there.txt")), "hi/there.txt");
        t(&(PosixPath("hi/there.txt")
           .with_filetype("")), "hi/there");

        t(&(PosixPath("/a/b/c/there.txt")
            .with_dirname("hi")), "hi/there.txt");

        t(&(PosixPath("hi/there.txt")
            .with_dirname(".")), "./there.txt");

        t(&(PosixPath("a/b/c")
            .push("..")), "a/b/c/..");

        t(&(PosixPath("there.txt")
            .with_filetype("o")), "there.o");

        t(&(PosixPath("hi/there.txt")
            .with_filetype("o")), "hi/there.o");

        t(&(PosixPath("hi/there.txt")
            .with_filetype("o")
            .with_dirname("/usr/lib")),
          "/usr/lib/there.o");

        t(&(PosixPath("hi/there.txt")
            .with_filetype("o")
            .with_dirname("/usr/lib/")),
          "/usr/lib/there.o");

        t(&(PosixPath("hi/there.txt")
            .with_filetype("o")
            .with_dirname("/usr//lib//")),
            "/usr/lib/there.o");

        t(&(PosixPath("/usr/bin/rust")
            .push_many([~"lib", ~"thingy.so"])
            .with_filestem("librustc")),
          "/usr/bin/rust/lib/librustc.so");

    }

    #[test]
    fn test_normalize() {
        fn t(wp: &PosixPath, s: &str) {
            let ss = wp.to_str();
            let sss = s.to_owned();
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert_eq!(ss, sss);
            }
        }

        t(&(PosixPath("hi/there.txt")
            .with_dirname(".").normalize()), "there.txt");

        t(&(PosixPath("a/b/../c/././/../foo.txt/").normalize()),
          "a/foo.txt");

        t(&(PosixPath("a/b/c")
            .push("..").normalize()), "a/b");
    }

    #[test]
    fn test_extract_unc_prefixes() {
        assert!(windows::extract_unc_prefix("\\\\").is_none());
        assert!(windows::extract_unc_prefix("//").is_none());
        assert!(windows::extract_unc_prefix("\\\\hi").is_none());
        assert!(windows::extract_unc_prefix("//hi").is_none());
        assert!(windows::extract_unc_prefix("\\\\hi\\") ==
            Some((~"hi", ~"\\")));
        assert!(windows::extract_unc_prefix("//hi\\") ==
            Some((~"hi", ~"\\")));
        assert!(windows::extract_unc_prefix("\\\\hi\\there") ==
            Some((~"hi", ~"\\there")));
        assert!(windows::extract_unc_prefix("//hi/there") ==
            Some((~"hi", ~"/there")));
        assert!(windows::extract_unc_prefix(
            "\\\\hi\\there\\friends.txt") ==
            Some((~"hi", ~"\\there\\friends.txt")));
        assert!(windows::extract_unc_prefix(
            "//hi\\there\\friends.txt") ==
            Some((~"hi", ~"\\there\\friends.txt")));
    }

    #[test]
    fn test_extract_drive_prefixes() {
        assert!(windows::extract_drive_prefix("c").is_none());
        assert!(windows::extract_drive_prefix("c:") ==
                     Some((~"c", ~"")));
        assert!(windows::extract_drive_prefix("d:") ==
                     Some((~"d", ~"")));
        assert!(windows::extract_drive_prefix("z:") ==
                     Some((~"z", ~"")));
        assert!(windows::extract_drive_prefix("c:\\hi") ==
                     Some((~"c", ~"\\hi")));
        assert!(windows::extract_drive_prefix("d:hi") ==
                     Some((~"d", ~"hi")));
        assert!(windows::extract_drive_prefix("c:hi\\there.txt") ==
                     Some((~"c", ~"hi\\there.txt")));
        assert!(windows::extract_drive_prefix("c:\\hi\\there.txt") ==
                     Some((~"c", ~"\\hi\\there.txt")));
    }

    #[test]
    fn test_windows_paths() {
        fn t(wp: &WindowsPath, s: &str) {
            let ss = wp.to_str();
            let sss = s.to_owned();
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert_eq!(ss, sss);
            }
        }

        t(&(WindowsPath("hi")), "hi");
        t(&(WindowsPath("hi/there")), "hi\\there");
        t(&(WindowsPath("hi/there.txt")), "hi\\there.txt");

        t(&(WindowsPath("there.txt")
            .with_filetype("o")), "there.o");

        t(&(WindowsPath("hi/there.txt")
            .with_filetype("o")), "hi\\there.o");

        t(&(WindowsPath("hi/there.txt")
            .with_filetype("o")
            .with_dirname("c:\\program files A")),
          "c:\\program files A\\there.o");

        t(&(WindowsPath("hi/there.txt")
            .with_filetype("o")
            .with_dirname("c:\\program files B\\")),
          "c:\\program files B\\there.o");

        t(&(WindowsPath("hi/there.txt")
            .with_filetype("o")
            .with_dirname("c:\\program files C\\/")),
            "c:\\program files C\\there.o");

        t(&(WindowsPath("c:\\program files (x86)\\rust")
            .push_many([~"lib", ~"thingy.dll"])
            .with_filename("librustc.dll")),
          "c:\\program files (x86)\\rust\\lib\\librustc.dll");

        t(&(WindowsPath("\\\\computer\\share")
            .unsafe_join(&WindowsPath("\\a"))),
          "\\\\computer\\a");

        t(&(WindowsPath("//computer/share")
            .unsafe_join(&WindowsPath("\\a"))),
          "\\\\computer\\a");

        t(&(WindowsPath("//computer/share")
            .unsafe_join(&WindowsPath("\\\\computer\\share"))),
          "\\\\computer\\share");

        t(&(WindowsPath("C:/whatever")
            .unsafe_join(&WindowsPath("//computer/share/a/b"))),
          "\\\\computer\\share\\a\\b");

        t(&(WindowsPath("C:")
            .unsafe_join(&WindowsPath("D:/foo"))),
          "D:\\foo");

        t(&(WindowsPath("C:")
            .unsafe_join(&WindowsPath("B"))),
          "C:B");

        t(&(WindowsPath("C:")
            .unsafe_join(&WindowsPath("/foo"))),
          "C:\\foo");

        t(&(WindowsPath("C:\\")
            .unsafe_join(&WindowsPath("\\bar"))),
          "C:\\bar");

        t(&(WindowsPath("")
            .unsafe_join(&WindowsPath(""))),
          "");

        t(&(WindowsPath("")
            .unsafe_join(&WindowsPath("a"))),
          "a");

        t(&(WindowsPath("")
            .unsafe_join(&WindowsPath("C:\\a"))),
          "C:\\a");

        t(&(WindowsPath("c:\\foo")
            .normalize()),
          "C:\\foo");
    }

    #[test]
    fn test_windows_path_restrictions() {
        assert_eq!(WindowsPath("hi").is_restricted(), false);
        assert_eq!(WindowsPath("C:\\NUL").is_restricted(), true);
        assert_eq!(WindowsPath("C:\\COM1.TXT").is_restricted(), true);
        assert_eq!(WindowsPath("c:\\prn.exe").is_restricted(), true);
    }
}
