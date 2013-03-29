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

use cmp::Eq;
use libc;
use option::{None, Option, Some};
use str;
use to_str::ToStr;

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
    fn from_str(&str) -> Self;

    fn dirname(&self) -> ~str;
    fn filename(&self) -> Option<~str>;
    fn filestem(&self) -> Option<~str>;
    fn filetype(&self) -> Option<~str>;

    fn with_dirname(&self, (&str)) -> Self;
    fn with_filename(&self, (&str)) -> Self;
    fn with_filestem(&self, (&str)) -> Self;
    fn with_filetype(&self, (&str)) -> Self;

    fn dir_path(&self) -> Self;
    fn file_path(&self) -> Self;

    fn push(&self, (&str)) -> Self;
    fn push_rel(&self, (&Self)) -> Self;
    fn push_many(&self, (&[~str])) -> Self;
    fn pop(&self) -> Self;

    fn unsafe_join(&self, (&Self)) -> Self;
    fn is_restricted(&self) -> bool;

    fn normalize(&self) -> Self;
}

#[cfg(windows)]
pub type Path = WindowsPath;

#[cfg(windows)]
pub fn Path(s: &str) -> Path {
    WindowsPath(s)
}

#[cfg(unix)]
pub type Path = PosixPath;

#[cfg(unix)]
pub fn Path(s: &str) -> Path {
    PosixPath(s)
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
mod stat {
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
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


pub impl Path {
    fn stat(&self) -> Option<libc::stat> {
        unsafe {
             do str::as_c_str(self.to_str()) |buf| {
                let mut st = stat::arch::default_stat();
                let r = libc::stat(buf, &mut st);

                if r == 0 { Some(st) } else { None }
            }
        }
    }

    #[cfg(unix)]
    fn lstat(&self) -> Option<libc::stat> {
        unsafe {
            do str::as_c_str(self.to_str()) |buf| {
                let mut st = stat::arch::default_stat();
                let r = libc::lstat(buf, &mut st);

                if r == 0 { Some(st) } else { None }
            }
        }
    }

    fn exists(&self) -> bool {
        match self.stat() {
            None => false,
            Some(_) => true,
        }
    }

    fn get_size(&self) -> Option<i64> {
        match self.stat() {
            None => None,
            Some(ref st) => Some(st.st_size as i64),
        }
    }

    fn get_mode(&self) -> Option<uint> {
        match self.stat() {
            None => None,
            Some(ref st) => Some(st.st_mode as uint),
        }
    }
}

#[cfg(target_os = "freebsd")]
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
pub impl Path {
    fn get_atime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_atime as i64,
                      st.st_atime_nsec as int))
            }
        }
    }

    fn get_mtime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_mtime as i64,
                      st.st_mtime_nsec as int))
            }
        }
    }

    fn get_ctime(&self) -> Option<(i64, int)> {
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
pub impl Path {
    fn get_birthtime(&self) -> Option<(i64, int)> {
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
pub impl Path {
    fn get_atime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_atime as i64, 0))
            }
        }
    }

    fn get_mtime(&self) -> Option<(i64, int)> {
        match self.stat() {
            None => None,
            Some(ref st) => {
                Some((st.st_mtime as i64, 0))
            }
        }
    }

    fn get_ctime(&self) -> Option<(i64, int)> {
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
        s + str::connect(self.components, "/")
    }
}

// FIXME (#3227): when default methods in traits are working, de-duplicate
// PosixPath and WindowsPath, most of their methods are common.
impl GenericPath for PosixPath {

    fn from_str(s: &str) -> PosixPath {
        let mut components = ~[];
        for str::each_split_nonempty(s, |c| c == '/') |s| { components.push(s.to_owned()) }
        let is_absolute = (s.len() != 0 && s[0] == '/' as u8);
        return PosixPath { is_absolute: is_absolute,
                           components: components }
    }

    fn dirname(&self) -> ~str {
        unsafe {
            let s = self.dir_path().to_str();
            if s.len() == 0 {
                ~"."
            } else {
                s
            }
        }
    }

    fn filename(&self) -> Option<~str> {
        match self.components.len() {
          0 => None,
          n => Some(copy self.components[n - 1])
        }
    }

    fn filestem(&self) -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) => Some(f.slice(0, p).to_owned()),
              None => Some(copy *f)
            }
          }
        }
    }

    fn filetype(&self) -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) if p < f.len() => Some(f.slice(p, f.len()).to_owned()),
              _ => None
            }
          }
        }
    }

    fn with_dirname(&self, d: &str) -> PosixPath {
        let dpath = PosixPath(d);
        match self.filename() {
          Some(ref f) => dpath.push(*f),
          None => dpath
        }
    }

    fn with_filename(&self, f: &str) -> PosixPath {
        unsafe {
            assert!(! str::any(f, |c| windows::is_sep(c as u8)));
            self.dir_path().push(f)
        }
    }

    fn with_filestem(&self, s: &str) -> PosixPath {
        match self.filetype() {
          None => self.with_filename(s),
          Some(ref t) => self.with_filename(str::from_slice(s) + *t)
        }
    }

    fn with_filetype(&self, t: &str) -> PosixPath {
        if t.len() == 0 {
            match self.filestem() {
              None => copy *self,
              Some(ref s) => self.with_filename(*s)
            }
        } else {
            let t = ~"." + str::from_slice(t);
            match self.filestem() {
              None => self.with_filename(t),
              Some(ref s) => self.with_filename(*s + t)
            }
        }
    }

    fn dir_path(&self) -> PosixPath {
        if self.components.len() != 0 {
            self.pop()
        } else {
            copy *self
        }
    }

    fn file_path(&self) -> PosixPath {
        let cs = match self.filename() {
          None => ~[],
          Some(ref f) => ~[copy *f]
        };
        return PosixPath { is_absolute: false,
                           components: cs }
    }

    fn push_rel(&self, other: &PosixPath) -> PosixPath {
        assert!(!other.is_absolute);
        self.push_many(other.components)
    }

    fn unsafe_join(&self, other: &PosixPath) -> PosixPath {
        if other.is_absolute {
            PosixPath { is_absolute: true,
                        components: copy other.components }
        } else {
            self.push_rel(other)
        }
    }

    fn is_restricted(&self) -> bool {
        false
    }

    fn push_many(&self, cs: &[~str]) -> PosixPath {
        let mut v = copy self.components;
        for cs.each |e| {
            let mut ss = ~[];
            for str::each_split_nonempty(*e, |c| windows::is_sep(c as u8)) |s| {
                ss.push(s.to_owned())
            }
            unsafe { v.push_all_move(ss); }
        }
        PosixPath { is_absolute: self.is_absolute,
                    components: v }
    }

    fn push(&self, s: &str) -> PosixPath {
        let mut v = copy self.components;
        let mut ss = ~[];
        for str::each_split_nonempty(s, |c| windows::is_sep(c as u8)) |s| {
            ss.push(s.to_owned())
        }
        unsafe { v.push_all_move(ss); }
        PosixPath { components: v, ..copy *self }
    }

    fn pop(&self) -> PosixPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            unsafe { cs.pop(); }
        }
        return PosixPath {
            is_absolute: self.is_absolute,
            components: cs
        }
                          //..self }
    }

    fn normalize(&self) -> PosixPath {
        return PosixPath {
            is_absolute: self.is_absolute,
            components: normalize(self.components)
          //  ..self
        }
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
        s + str::connect(self.components, "\\")
    }
}


impl GenericPath for WindowsPath {

    fn from_str(s: &str) -> WindowsPath {
        let host;
        let device;
        let rest;

        match windows::extract_drive_prefix(s) {
          Some((ref d, ref r)) => {
            host = None;
            device = Some(copy *d);
            rest = copy *r;
          }
          None => {
            match windows::extract_unc_prefix(s) {
              Some((ref h, ref r)) => {
                host = Some(copy *h);
                device = None;
                rest = copy *r;
              }
              None => {
                host = None;
                device = None;
                rest = str::from_slice(s);
              }
            }
          }
        }

        let mut components = ~[];
        for str::each_split_nonempty(rest, |c| windows::is_sep(c as u8)) |s| {
            components.push(s.to_owned())
        }
        let is_absolute = (rest.len() != 0 && windows::is_sep(rest[0]));
        return WindowsPath { host: host,
                             device: device,
                             is_absolute: is_absolute,
                             components: components }
    }

    fn dirname(&self) -> ~str {
        unsafe {
            let s = self.dir_path().to_str();
            if s.len() == 0 {
                ~"."
            } else {
                s
            }
        }
    }

    fn filename(&self) -> Option<~str> {
        match self.components.len() {
          0 => None,
          n => Some(copy self.components[n - 1])
        }
    }

    fn filestem(&self) -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) => Some(f.slice(0, p).to_owned()),
              None => Some(copy *f)
            }
          }
        }
    }

    fn filetype(&self) -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) if p < f.len() => Some(f.slice(p, f.len()).to_owned()),
              _ => None
            }
          }
        }
    }

    fn with_dirname(&self, d: &str) -> WindowsPath {
        let dpath = WindowsPath(d);
        match self.filename() {
          Some(ref f) => dpath.push(*f),
          None => dpath
        }
    }

    fn with_filename(&self, f: &str) -> WindowsPath {
        assert!(! str::any(f, |c| windows::is_sep(c as u8)));
        self.dir_path().push(f)
    }

    fn with_filestem(&self, s: &str) -> WindowsPath {
        match self.filetype() {
          None => self.with_filename(s),
          Some(ref t) => self.with_filename(str::from_slice(s) + *t)
        }
    }

    fn with_filetype(&self, t: &str) -> WindowsPath {
        if t.len() == 0 {
            match self.filestem() {
              None => copy *self,
              Some(ref s) => self.with_filename(*s)
            }
        } else {
            let t = ~"." + str::from_slice(t);
            match self.filestem() {
              None => self.with_filename(t),
              Some(ref s) =>
              self.with_filename(*s + t)
            }
        }
    }

    fn dir_path(&self) -> WindowsPath {
        if self.components.len() != 0 {
            self.pop()
        } else {
            copy *self
        }
    }

    fn file_path(&self) -> WindowsPath {
        let cs = match self.filename() {
          None => ~[],
          Some(ref f) => ~[copy *f]
        };
        return WindowsPath { host: None,
                             device: None,
                             is_absolute: false,
                             components: cs }
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
            Some(copy host) => {
                return WindowsPath {
                    host: Some(host),
                    device: copy other.device,
                    is_absolute: true,
                    components: copy other.components
                };
            }
            _ => {}
        }

        /* if rhs has a device set, then a part wins */
        match other.device {
            Some(copy device) => {
                return WindowsPath {
                    host: None,
                    device: Some(device),
                    is_absolute: true,
                    components: copy other.components
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
            components: copy other.components
        }
    }

    fn is_restricted(&self) -> bool {
        match self.filestem() {
            Some(stem) => {
                match stem.to_lower() {
                    ~"con" | ~"aux" | ~"com1" | ~"com2" | ~"com3" | ~"com4" |
                    ~"lpt1" | ~"lpt2" | ~"lpt3" | ~"prn" | ~"nul" => true,
                    _ => false
                }
            },
            None => false
        }
    }

    fn push_many(&self, cs: &[~str]) -> WindowsPath {
        let mut v = copy self.components;
        for cs.each |e| {
            let mut ss = ~[];
            for str::each_split_nonempty(*e, |c| windows::is_sep(c as u8)) |s| {
                ss.push(s.to_owned())
            }
            unsafe { v.push_all_move(ss); }
        }
        // tedious, but as-is, we can't use ..self
        return WindowsPath {
            host: copy self.host,
            device: copy self.device,
            is_absolute: self.is_absolute,
            components: v
        }
    }

    fn push(&self, s: &str) -> WindowsPath {
        let mut v = copy self.components;
        let mut ss = ~[];
        for str::each_split_nonempty(s, |c| windows::is_sep(c as u8)) |s| {
            ss.push(s.to_owned())
        }
        unsafe { v.push_all_move(ss); }
        return WindowsPath { components: v, ..copy *self }
    }

    fn pop(&self) -> WindowsPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            unsafe { cs.pop(); }
        }
        return WindowsPath {
            host: copy self.host,
            device: copy self.device,
            is_absolute: self.is_absolute,
            components: cs
        }
    }

    fn normalize(&self) -> WindowsPath {
        return WindowsPath {
            host: copy self.host,
            device: match self.device {
                None => None,
                Some(ref device) => Some(device.to_upper())
            },
            is_absolute: self.is_absolute,
            components: normalize(self.components)
        }
    }
}


pub fn normalize(components: &[~str]) -> ~[~str] {
    let mut cs = ~[];
    unsafe {
        for components.each |c| {
            unsafe {
                if *c == ~"." && components.len() > 1 { loop; }
                if *c == ~"" { loop; }
                if *c == ~".." && cs.len() != 0 {
                    cs.pop();
                    loop;
                }
                cs.push(copy *c);
            }
        }
    }
    cs
}

// Various windows helpers, and tests for the impl.
pub mod windows {
    use libc;
    use option::{None, Option, Some};

    #[inline(always)]
    pub fn is_sep(u: u8) -> bool {
        u == '/' as u8 || u == '\\' as u8
    }

    pub fn extract_unc_prefix(s: &str) -> Option<(~str,~str)> {
        if (s.len() > 1 &&
            (s[0] == '\\' as u8 || s[0] == '/' as u8) &&
            s[0] == s[1]) {
            let mut i = 2;
            while i < s.len() {
                if is_sep(s[i]) {
                    let pre = s.slice(2, i).to_owned();
                    let mut rest = s.slice(i, s.len()).to_owned();
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
    use str;

    #[test]
    fn test_double_slash_collapsing() {
        let path = PosixPath("tmp/");
        let path = path.push("/hmm");
        let path = path.normalize();
        assert!(~"tmp/hmm" == path.to_str());

        let path = WindowsPath("tmp/");
        let path = path.push("/hmm");
        let path = path.normalize();
        assert!(~"tmp\\hmm" == path.to_str());
    }

    #[test]
    fn test_filetype_foo_bar() {
        let wp = PosixPath("foo.bar");
        assert!(wp.filetype() == Some(~".bar"));

        let wp = WindowsPath("foo.bar");
        assert!(wp.filetype() == Some(~".bar"));
    }

    #[test]
    fn test_filetype_foo() {
        let wp = PosixPath("foo");
        assert!(wp.filetype() == None);

        let wp = WindowsPath("foo");
        assert!(wp.filetype() == None);
    }

    #[test]
    fn test_posix_paths() {
        fn t(wp: &PosixPath, s: &str) {
            let ss = wp.to_str();
            let sss = str::from_slice(s);
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert!(ss == sss);
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
            let sss = str::from_slice(s);
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert!(ss == sss);
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
            let sss = str::from_slice(s);
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert!(ss == sss);
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
        assert!(WindowsPath("hi").is_restricted() == false);
        assert!(WindowsPath("C:\\NUL").is_restricted() == true);
        assert!(WindowsPath("C:\\COM1.TXT").is_restricted() == true);
        assert!(WindowsPath("c:\\prn.exe").is_restricted() == true);
    }
}
