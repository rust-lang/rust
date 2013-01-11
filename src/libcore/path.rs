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

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::Eq;
use libc;
use option::{None, Option, Some};
use ptr;
use str;
use to_str::ToStr;

#[deriving_eq]
pub struct WindowsPath {
    host: Option<~str>,
    device: Option<~str>,
    is_absolute: bool,
    components: ~[~str],
}

pub pure fn WindowsPath(s: &str) -> WindowsPath {
    GenericPath::from_str(s)
}

#[deriving_eq]
pub struct PosixPath {
    is_absolute: bool,
    components: ~[~str],
}

pub pure fn PosixPath(s: &str) -> PosixPath {
    GenericPath::from_str(s)
}

pub trait GenericPath {

    static pure fn from_str(&str) -> self;

    pure fn dirname() -> ~str;
    pure fn filename() -> Option<~str>;
    pure fn filestem() -> Option<~str>;
    pure fn filetype() -> Option<~str>;

    pure fn with_dirname((&str)) -> self;
    pure fn with_filename((&str)) -> self;
    pure fn with_filestem((&str)) -> self;
    pure fn with_filetype((&str)) -> self;

    pure fn dir_path() -> self;
    pure fn file_path() -> self;

    pure fn push((&str)) -> self;
    pure fn push_rel((&self)) -> self;
    pure fn push_many((&[~str])) -> self;
    pure fn pop() -> self;

    pure fn normalize() -> self;
}

#[cfg(windows)]
pub type Path = WindowsPath;

#[cfg(windows)]
pub pure fn Path(s: &str) -> Path {
    WindowsPath(s)
}

#[cfg(unix)]
pub type Path = PosixPath;

#[cfg(unix)]
pub pure fn Path(s: &str) -> Path {
    PosixPath(s)
}

#[cfg(target_os = "linux")]
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
    fn stat(&self) -> Option<libc::stat> {
         do str::as_c_str(self.to_str()) |buf| {
            let mut st = stat::arch::default_stat();
            let r = libc::stat(buf, ptr::mut_addr_of(&st));

            if r == 0 { Some(move st) } else { None }
        }
    }

    #[cfg(unix)]
    fn lstat(&self) -> Option<libc::stat> {
         do str::as_c_str(self.to_str()) |buf| {
            let mut st = stat::arch::default_stat();
            let r = libc::lstat(buf, ptr::mut_addr_of(&st));

            if r == 0 { Some(move st) } else { None }
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
impl Path {
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
impl Path {
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
impl Path {
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

impl PosixPath : ToStr {
    pure fn to_str() -> ~str {
        let mut s = ~"";
        if self.is_absolute {
            s += "/";
        }
        s + str::connect(self.components, "/")
    }
}

// FIXME (#3227): when default methods in traits are working, de-duplicate
// PosixPath and WindowsPath, most of their methods are common.
impl PosixPath : GenericPath {

    static pure fn from_str(s: &str) -> PosixPath {
        let mut components = str::split_nonempty(s, |c| c == '/');
        let is_absolute = (s.len() != 0 && s[0] == '/' as u8);
        return PosixPath { is_absolute: is_absolute,
                           components: move components }
    }

    pure fn dirname() -> ~str {
        unsafe {
            let s = self.dir_path().to_str();
            if s.len() == 0 {
                ~"."
            } else {
                move s
            }
        }
    }

    pure fn filename() -> Option<~str> {
        match self.components.len() {
          0 => None,
          n => Some(copy self.components[n - 1])
        }
    }

    pure fn filestem() -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) => Some(f.slice(0, p)),
              None => Some(copy *f)
            }
          }
        }
    }

    pure fn filetype() -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) if p < f.len() => Some(f.slice(p, f.len())),
              _ => None
            }
          }
        }
    }

    pure fn with_dirname(d: &str) -> PosixPath {
        let dpath = PosixPath(d);
        match self.filename() {
          Some(ref f) => dpath.push(*f),
          None => move dpath
        }
    }

    pure fn with_filename(f: &str) -> PosixPath {
        unsafe {
            assert ! str::any(f, |c| windows::is_sep(c as u8));
            self.dir_path().push(f)
        }
    }

    pure fn with_filestem(s: &str) -> PosixPath {
        match self.filetype() {
          None => self.with_filename(s),
          Some(ref t) => self.with_filename(str::from_slice(s) + *t)
        }
    }

    pure fn with_filetype(t: &str) -> PosixPath {
        if t.len() == 0 {
            match self.filestem() {
              None => copy self,
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

    pure fn dir_path() -> PosixPath {
        if self.components.len() != 0 {
            self.pop()
        } else {
            copy self
        }
    }

    pure fn file_path() -> PosixPath {
        let cs = match self.filename() {
          None => ~[],
          Some(ref f) => ~[copy *f]
        };
        return PosixPath { is_absolute: false,
                           components: move cs }
    }

    pure fn push_rel(other: &PosixPath) -> PosixPath {
        assert !other.is_absolute;
        self.push_many(other.components)
    }

    pure fn push_many(cs: &[~str]) -> PosixPath {
        let mut v = copy self.components;
        for cs.each |e| {
            let mut ss = str::split_nonempty(
                *e,
                |c| windows::is_sep(c as u8));
            unsafe { v.push_all_move(move ss); }
        }
        PosixPath { is_absolute: self.is_absolute,
                    components: move v }
    }

    pure fn push(s: &str) -> PosixPath {
        let mut v = copy self.components;
        let mut ss = str::split_nonempty(s, |c| windows::is_sep(c as u8));
        unsafe { v.push_all_move(move ss); }
        PosixPath { components: move v, ..copy self }
    }

    pure fn pop() -> PosixPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            unsafe { cs.pop(); }
        }
        return PosixPath {
            is_absolute: self.is_absolute,
            components: move cs
        }
                          //..self }
    }

    pure fn normalize() -> PosixPath {
        return PosixPath {
            is_absolute: self.is_absolute,
            components: normalize(self.components)
          //  ..self
        }
    }
}


impl WindowsPath : ToStr {
    pure fn to_str() -> ~str {
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


impl WindowsPath : GenericPath {

    static pure fn from_str(s: &str) -> WindowsPath {
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

        let mut components =
            str::split_nonempty(rest, |c| windows::is_sep(c as u8));
        let is_absolute = (rest.len() != 0 && windows::is_sep(rest[0]));
        return WindowsPath { host: move host,
                             device: move device,
                             is_absolute: is_absolute,
                             components: move components }
    }

    pure fn dirname() -> ~str {
        unsafe {
            let s = self.dir_path().to_str();
            if s.len() == 0 {
                ~"."
            } else {
                move s
            }
        }
    }

    pure fn filename() -> Option<~str> {
        match self.components.len() {
          0 => None,
          n => Some(copy self.components[n - 1])
        }
    }

    pure fn filestem() -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) => Some(f.slice(0, p)),
              None => Some(copy *f)
            }
          }
        }
    }

    pure fn filetype() -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) if p < f.len() => Some(f.slice(p, f.len())),
              _ => None
            }
          }
        }
    }

    pure fn with_dirname(d: &str) -> WindowsPath {
        let dpath = WindowsPath(d);
        match self.filename() {
          Some(ref f) => dpath.push(*f),
          None => move dpath
        }
    }

    pure fn with_filename(f: &str) -> WindowsPath {
        assert ! str::any(f, |c| windows::is_sep(c as u8));
        self.dir_path().push(f)
    }

    pure fn with_filestem(s: &str) -> WindowsPath {
        match self.filetype() {
          None => self.with_filename(s),
          Some(ref t) => self.with_filename(str::from_slice(s) + *t)
        }
    }

    pure fn with_filetype(t: &str) -> WindowsPath {
        if t.len() == 0 {
            match self.filestem() {
              None => copy self,
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

    pure fn dir_path() -> WindowsPath {
        if self.components.len() != 0 {
            self.pop()
        } else {
            copy self
        }
    }

    pure fn file_path() -> WindowsPath {
        let cs = match self.filename() {
          None => ~[],
          Some(ref f) => ~[copy *f]
        };
        return WindowsPath { host: None,
                             device: None,
                             is_absolute: false,
                             components: move cs }
    }

    pure fn push_rel(other: &WindowsPath) -> WindowsPath {
        assert !other.is_absolute;
        self.push_many(other.components)
    }

    pure fn push_many(cs: &[~str]) -> WindowsPath {
        let mut v = copy self.components;
        for cs.each |e| {
            let mut ss = str::split_nonempty(
                *e,
                |c| windows::is_sep(c as u8));
            unsafe { v.push_all_move(move ss); }
        }
        // tedious, but as-is, we can't use ..self
        return WindowsPath {
            host: copy self.host,
            device: copy self.device,
            is_absolute: self.is_absolute,
            components: move v
        }
    }

    pure fn push(s: &str) -> WindowsPath {
        let mut v = copy self.components;
        let mut ss = str::split_nonempty(s, |c| windows::is_sep(c as u8));
        unsafe { v.push_all_move(move ss); }
        return WindowsPath { components: move v, ..copy self }
    }

    pure fn pop() -> WindowsPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            unsafe { cs.pop(); }
        }
        return WindowsPath {
            host: copy self.host,
            device: copy self.device,
            is_absolute: self.is_absolute,
            components: move cs
        }
    }

    pure fn normalize() -> WindowsPath {
        return WindowsPath {
            host: copy self.host,
            device: copy self.device,
            is_absolute: self.is_absolute,
            components: normalize(self.components)
        }
    }
}


pub pure fn normalize(components: &[~str]) -> ~[~str] {
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
    move cs
}

// Various windows helpers, and tests for the impl.
pub mod windows {
    use libc;
    use option::{None, Option, Some};
    use to_str::ToStr;

    #[inline(always)]
    pub pure fn is_sep(u: u8) -> bool {
        u == '/' as u8 || u == '\\' as u8
    }

    pub pure fn extract_unc_prefix(s: &str) -> Option<(~str,~str)> {
        if (s.len() > 1 &&
            s[0] == '\\' as u8 &&
            s[1] == '\\' as u8) {
            let mut i = 2;
            while i < s.len() {
                if s[i] == '\\' as u8 {
                    let pre = s.slice(2, i);
                    let rest = s.slice(i, s.len());
                    return Some((move pre, move rest));
                }
                i += 1;
            }
        }
        None
    }

    pub pure fn extract_drive_prefix(s: &str) -> Option<(~str,~str)> {
        unsafe {
            if (s.len() > 1 &&
                libc::isalpha(s[0] as libc::c_int) != 0 &&
                s[1] == ':' as u8) {
                let rest = if s.len() == 2 {
                    ~""
                } else {
                    s.slice(2, s.len())
                };
                return Some((s.slice(0,1), move rest));
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
        assert ~"tmp/hmm" == path.to_str();

        let path = WindowsPath("tmp/");
        let path = path.push("/hmm");
        let path = path.normalize();
        assert ~"tmp\\hmm" == path.to_str();
    }

    #[test]
    fn test_filetype_foo_bar() {
        let wp = PosixPath("foo.bar");
        assert wp.filetype() == Some(~".bar");

        let wp = WindowsPath("foo.bar");
        assert wp.filetype() == Some(~".bar");
    }

    #[test]
    fn test_filetype_foo() {
        let wp = PosixPath("foo");
        assert wp.filetype() == None;

        let wp = WindowsPath("foo");
        assert wp.filetype() == None;
    }

    #[test]
    fn test_posix_paths() {
        fn t(wp: &PosixPath, s: &str) {
            let ss = wp.to_str();
            let sss = str::from_slice(s);
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert ss == sss;
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
                assert ss == sss;
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
        assert windows::extract_unc_prefix("\\\\").is_none();
        assert windows::extract_unc_prefix("\\\\hi").is_none();
        assert windows::extract_unc_prefix("\\\\hi\\") ==
            Some((~"hi", ~"\\"));
        assert windows::extract_unc_prefix("\\\\hi\\there") ==
            Some((~"hi", ~"\\there"));
        assert windows::extract_unc_prefix("\\\\hi\\there\\friends.txt") ==
            Some((~"hi", ~"\\there\\friends.txt"));
    }

    #[test]
    fn test_extract_drive_prefixes() {
        assert windows::extract_drive_prefix("c").is_none();
        assert windows::extract_drive_prefix("c:") == Some((~"c", ~""));
        assert windows::extract_drive_prefix("d:") == Some((~"d", ~""));
        assert windows::extract_drive_prefix("z:") == Some((~"z", ~""));
        assert windows::extract_drive_prefix("c:\\hi") ==
            Some((~"c", ~"\\hi"));
        assert windows::extract_drive_prefix("d:hi") == Some((~"d", ~"hi"));
        assert windows::extract_drive_prefix("c:hi\\there.txt") ==
            Some((~"c", ~"hi\\there.txt"));
        assert windows::extract_drive_prefix("c:\\hi\\there.txt") ==
            Some((~"c", ~"\\hi\\there.txt"));
    }

    #[test]
    fn test_windows_paths() {
        fn t(wp: &WindowsPath, s: &str) {
            let ss = wp.to_str();
            let sss = str::from_slice(s);
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert ss == sss;
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
    }
}
