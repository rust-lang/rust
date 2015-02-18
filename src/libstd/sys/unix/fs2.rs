// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;
use io::prelude::*;
use os::unix::prelude::*;

use ffi::{CString, CStr, OsString, AsOsStr, OsStr};
use io::{self, Error, Seek, SeekFrom};
use libc::{self, c_int, c_void, size_t, off_t, c_char, mode_t};
use mem;
use path::{Path, PathBuf};
use ptr;
use rc::Rc;
use sys::fd::FileDesc;
use sys::{c, cvt, cvt_r};
use sys_common::FromInner;
use vec::Vec;

pub struct File(FileDesc);

pub struct FileAttr {
    stat: libc::stat,
}

pub struct ReadDir {
    dirp: *mut libc::DIR,
    root: Rc<PathBuf>,
}

pub struct DirEntry {
    buf: Vec<u8>,
    dirent: *mut libc::dirent_t,
    root: Rc<PathBuf>,
}

#[derive(Clone)]
pub struct OpenOptions {
    flags: c_int,
    read: bool,
    write: bool,
    mode: mode_t,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions { mode: mode_t }

impl FileAttr {
    pub fn is_dir(&self) -> bool {
        (self.stat.st_mode as mode_t) & libc::S_IFMT == libc::S_IFDIR
    }
    pub fn is_file(&self) -> bool {
        (self.stat.st_mode as mode_t) & libc::S_IFMT == libc::S_IFREG
    }
    pub fn size(&self) -> u64 { self.stat.st_size as u64 }
    pub fn perm(&self) -> FilePermissions {
        FilePermissions { mode: (self.stat.st_mode as mode_t) & 0o777 }
    }

    pub fn accessed(&self) -> u64 {
        self.mktime(self.stat.st_atime as u64, self.stat.st_atime_nsec as u64)
    }
    pub fn modified(&self) -> u64 {
        self.mktime(self.stat.st_mtime as u64, self.stat.st_mtime_nsec as u64)
    }

    // times are in milliseconds (currently)
    fn mktime(&self, secs: u64, nsecs: u64) -> u64 {
        secs * 1000 + nsecs / 1000000
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool { self.mode & 0o222 == 0 }
    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.mode &= !0o222;
        } else {
            self.mode |= 0o222;
        }
    }
}

impl FromInner<i32> for FilePermissions {
    fn from_inner(mode: i32) -> FilePermissions {
        FilePermissions { mode: mode as mode_t }
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        extern {
            fn rust_dirent_t_size() -> c_int;
        }

        let mut buf: Vec<u8> = Vec::with_capacity(unsafe {
            rust_dirent_t_size() as usize
        });
        let ptr = buf.as_mut_ptr() as *mut libc::dirent_t;

        let mut entry_ptr = ptr::null_mut();
        loop {
            if unsafe { libc::readdir_r(self.dirp, ptr, &mut entry_ptr) != 0 } {
                return Some(Err(Error::last_os_error()))
            }
            if entry_ptr.is_null() {
                return None
            }

            let entry = DirEntry {
                buf: buf,
                dirent: entry_ptr,
                root: self.root.clone()
            };
            if entry.name_bytes() == b"." || entry.name_bytes() == b".." {
                buf = entry.buf;
            } else {
                return Some(Ok(entry))
            }
        }
    }
}

impl Drop for ReadDir {
    fn drop(&mut self) {
        let r = unsafe { libc::closedir(self.dirp) };
        debug_assert_eq!(r, 0);
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.root.join(<OsStr as OsStrExt>::from_bytes(self.name_bytes()))
    }

    fn name_bytes(&self) -> &[u8] {
        extern {
            fn rust_list_dir_val(ptr: *mut libc::dirent_t) -> *const c_char;
        }
        unsafe {
            CStr::from_ptr(rust_list_dir_val(self.dirent)).to_bytes()
        }
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            flags: 0,
            read: false,
            write: false,
            mode: 0o666,
        }
    }

    pub fn read(&mut self, read: bool) {
        self.read = read;
    }

    pub fn write(&mut self, write: bool) {
        self.write = write;
    }

    pub fn append(&mut self, append: bool) {
        self.flag(libc::O_APPEND, append);
    }

    pub fn truncate(&mut self, truncate: bool) {
        self.flag(libc::O_TRUNC, truncate);
    }

    pub fn create(&mut self, create: bool) {
        self.flag(libc::O_CREAT, create);
    }

    pub fn mode(&mut self, mode: i32) {
        self.mode = mode as mode_t;
    }

    fn flag(&mut self, bit: c_int, on: bool) {
        if on {
            self.flags |= bit;
        } else {
            self.flags &= !bit;
        }
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let flags = opts.flags | match (opts.read, opts.write) {
            (true, true) => libc::O_RDWR,
            (false, true) => libc::O_WRONLY,
            (true, false) |
            (false, false) => libc::O_RDONLY,
        };
        let path = try!(cstr(path));
        let fd = try!(cvt_r(|| unsafe {
            libc::open(path.as_ptr(), flags, opts.mode)
        }));
        Ok(File(FileDesc::new(fd)))
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let mut stat: libc::stat = unsafe { mem::zeroed() };
        try!(cvt(unsafe { libc::fstat(self.0.raw(), &mut stat) }));
        Ok(FileAttr { stat: stat })
    }

    pub fn fsync(&self) -> io::Result<()> {
        try!(cvt_r(|| unsafe { libc::fsync(self.0.raw()) }));
        Ok(())
    }

    pub fn datasync(&self) -> io::Result<()> {
        try!(cvt_r(|| unsafe { os_datasync(self.0.raw()) }));
        return Ok(());

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        unsafe fn os_datasync(fd: c_int) -> c_int {
            libc::fcntl(fd, libc::F_FULLFSYNC)
        }
        #[cfg(target_os = "linux")]
        unsafe fn os_datasync(fd: c_int) -> c_int { libc::fdatasync(fd) }
        #[cfg(not(any(target_os = "macos",
                      target_os = "ios",
                      target_os = "linux")))]
        unsafe fn os_datasync(fd: c_int) -> c_int { libc::fsync(fd) }
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        try!(cvt_r(|| unsafe {
            libc::ftruncate(self.0.raw(), size as libc::off_t)
        }));
        Ok(())
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn flush(&self) -> io::Result<()> { Ok(()) }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            SeekFrom::Start(off) => (libc::SEEK_SET, off as off_t),
            SeekFrom::End(off) => (libc::SEEK_END, off as off_t),
            SeekFrom::Current(off) => (libc::SEEK_CUR, off as off_t),
        };
        let n = try!(cvt(unsafe { libc::lseek(self.0.raw(), pos, whence) }));
        Ok(n as u64)
    }

    pub fn fd(&self) -> &FileDesc { &self.0 }
}

fn cstr(path: &Path) -> io::Result<CString> {
    let cstring = try!(path.as_os_str().to_cstring());
    Ok(cstring)
}

pub fn mkdir(p: &Path) -> io::Result<()> {
    let p = try!(cstr(p));
    try!(cvt(unsafe { libc::mkdir(p.as_ptr(), 0o777) }));
    Ok(())
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let root = Rc::new(p.to_path_buf());
    let p = try!(cstr(p));
    unsafe {
        let ptr = libc::opendir(p.as_ptr());
        if ptr.is_null() {
            Err(Error::last_os_error())
        } else {
            Ok(ReadDir { dirp: ptr, root: root })
        }
    }
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let p = try!(cstr(p));
    try!(cvt(unsafe { libc::unlink(p.as_ptr()) }));
    Ok(())
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let old = try!(cstr(old));
    let new = try!(cstr(new));
    try!(cvt(unsafe { libc::rename(old.as_ptr(), new.as_ptr()) }));
    Ok(())
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    let p = try!(cstr(p));
    try!(cvt_r(|| unsafe { libc::chmod(p.as_ptr(), perm.mode) }));
    Ok(())
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let p = try!(cstr(p));
    try!(cvt(unsafe { libc::rmdir(p.as_ptr()) }));
    Ok(())
}

pub fn chown(p: &Path, uid: isize, gid: isize) -> io::Result<()> {
    let p = try!(cstr(p));
    try!(cvt_r(|| unsafe {
        libc::chown(p.as_ptr(), uid as libc::uid_t, gid as libc::gid_t)
    }));
    Ok(())
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    let c_path = try!(cstr(p));
    let p = c_path.as_ptr();
    let mut len = unsafe { libc::pathconf(p as *mut _, libc::_PC_NAME_MAX) };
    if len < 0 {
        len = 1024; // FIXME: read PATH_MAX from C ffi?
    }
    let mut buf: Vec<u8> = Vec::with_capacity(len as usize);
    unsafe {
        let n = try!(cvt({
            libc::readlink(p, buf.as_ptr() as *mut c_char, len as size_t)
        }));
        buf.set_len(n as usize);
    }
    let s: OsString = OsStringExt::from_vec(buf);
    Ok(PathBuf::new(&s))
}

pub fn symlink(src: &Path, dst: &Path) -> io::Result<()> {
    let src = try!(cstr(src));
    let dst = try!(cstr(dst));
    try!(cvt(unsafe { libc::symlink(src.as_ptr(), dst.as_ptr()) }));
    Ok(())
}

pub fn link(src: &Path, dst: &Path) -> io::Result<()> {
    let src = try!(cstr(src));
    let dst = try!(cstr(dst));
    try!(cvt(unsafe { libc::link(src.as_ptr(), dst.as_ptr()) }));
    Ok(())
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let p = try!(cstr(p));
    let mut stat: libc::stat = unsafe { mem::zeroed() };
    try!(cvt(unsafe { libc::stat(p.as_ptr(), &mut stat) }));
    Ok(FileAttr { stat: stat })
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    let p = try!(cstr(p));
    let mut stat: libc::stat = unsafe { mem::zeroed() };
    try!(cvt(unsafe { libc::lstat(p.as_ptr(), &mut stat) }));
    Ok(FileAttr { stat: stat })
}

pub fn utimes(p: &Path, atime: u64, mtime: u64) -> io::Result<()> {
    let p = try!(cstr(p));
    let buf = [super::ms_to_timeval(atime), super::ms_to_timeval(mtime)];
    try!(cvt(unsafe { c::utimes(p.as_ptr(), buf.as_ptr()) }));
    Ok(())
}
