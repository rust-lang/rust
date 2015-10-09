// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use error::prelude::*;
use inner::prelude::*;
use os_str::prelude::*;
use io::prelude::*;

use fs as sys;
use c_str::{CString, CStr, NulError};
use core::fmt;
use libc::{self, c_int, size_t, off_t, c_char, mode_t};
use core::mem;
use core::result;
use core::ptr;
use alloc::arc::Arc;
use unix::fd::FileDesc;
use unix::platform::raw;
use unix::c;
use unix::cvt_r;
use collections::Vec;
use collections::borrow::ToOwned;

pub struct File(FileDesc);

pub struct FileAttr {
    stat: raw::stat,
}

pub struct ReadDir {
    dirp: Dir,
    root: Arc<OsString>,
}

struct Dir(*mut libc::DIR);

unsafe impl Send for Dir {}
unsafe impl Sync for Dir {}

pub struct DirEntry {
    buf: Vec<u8>, // actually *mut libc::dirent_t
    root: Arc<OsString>,
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

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct FileType { mode: mode_t }

pub struct DirBuilder { mode: mode_t }

impl sys::FileAttr<Fs> for FileAttr {
    fn size(&self) -> u64 { self.stat.st_size as u64 }
    fn perm(&self) -> FilePermissions {
        FilePermissions { mode: (self.stat.st_mode as mode_t) & 0o777 }
    }

    fn file_type(&self) -> FileType {
        FileType { mode: self.stat.st_mode as mode_t }
    }
}

impl AsInner<raw::stat> for FileAttr {
    fn as_inner(&self) -> &raw::stat { &self.stat }
}

impl sys::FilePermissions<Fs> for FilePermissions {
    fn readonly(&self) -> bool { self.mode & 0o222 == 0 }
    fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.mode &= !0o222;
        } else {
            self.mode |= 0o222;
        }
    }
    fn mode(&self) -> raw::mode_t { self.mode }
}

impl sys::FileType<Fs> for FileType {
    fn is_dir(&self) -> bool { self.is(libc::S_IFDIR) }
    fn is_file(&self) -> bool { self.is(libc::S_IFREG) }
    fn is_symlink(&self) -> bool { self.is(libc::S_IFLNK) }

    fn is(&self, mode: mode_t) -> bool { self.mode & libc::S_IFMT == mode }
}

impl FromInner<raw::mode_t> for FilePermissions {
    fn from_inner(mode: raw::mode_t) -> FilePermissions {
        FilePermissions { mode: mode as mode_t }
    }
}

impl sys::ReadDir<Fs> for ReadDir { }

impl Iterator for ReadDir {
    type Item = Result<DirEntry>;

    fn next(&mut self) -> Option<Result<DirEntry>> {
        extern {
            fn rust_dirent_t_size() -> c_int;
        }

        let mut buf: Vec<u8> = Vec::with_capacity(unsafe {
            rust_dirent_t_size() as usize
        });
        let ptr = buf.as_mut_ptr() as *mut libc::dirent_t;

        let mut entry_ptr = ptr::null_mut();
        loop {
            if unsafe { libc::readdir_r(self.dirp.0, ptr, &mut entry_ptr) != 0 } {
                return Some(Error::expect_last_result())
            }
            if entry_ptr.is_null() {
                return None
            }

            let entry = DirEntry {
                buf: buf,
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

impl Drop for Dir {
    fn drop(&mut self) {
        let r = unsafe { libc::closedir(self.0) };
        debug_assert_eq!(r, 0);
    }
}

impl sys::DirEntry<Fs> for DirEntry {
    fn file_name(&self) -> &OsStr {
        OsStr::from_bytes(self.name_bytes())
    }

    fn root(&self) -> &OsStr {
        &self.root
    }

    fn metadata(&self) -> Result<FileAttr> {
        lstat(&self.path())
    }

    fn file_type(&self) -> Result<FileType> {
        extern {
            fn rust_dir_get_mode(ptr: *mut libc::dirent_t) -> c_int;
        }
        unsafe {
            match rust_dir_get_mode(self.dirent()) {
                -1 => lstat(&self.path()).map(|m| sys::FileAttr::file_type(&m)),
                n => Ok(FileType { mode: n as mode_t }),
            }
        }
    }

    fn ino(&self) -> raw::ino_t {
        extern {
            fn rust_dir_get_ino(ptr: *mut libc::dirent_t) -> raw::ino_t;
        }
        unsafe { rust_dir_get_ino(self.dirent()) }
    }
}

impl DirEntry {
    fn name_bytes(&self) -> &[u8] {
        extern {
            fn rust_list_dir_val(ptr: *mut libc::dirent_t) -> *const c_char;
        }
        unsafe {
            CStr::from_ptr(rust_list_dir_val(self.dirent())).to_bytes()
        }
    }

    fn dirent(&self) -> *mut libc::dirent_t {
        self.buf.as_ptr() as *mut _
    }

    fn path(&self) -> OsString {
        let mut path = (*self.root).clone();
        path.path_join(sys::DirEntry::file_name(self));
        path
    }
}

impl sys::OpenOptions<Fs> for OpenOptions {
    fn new() -> OpenOptions {
        OpenOptions {
            flags: libc::O_CLOEXEC,
            read: false,
            write: false,
            mode: 0o666,
        }
    }

    fn read(&mut self, read: bool) {
        self.read = read;
    }

    fn write(&mut self, write: bool) {
        self.write = write;
    }

    fn append(&mut self, append: bool) {
        self.flag(libc::O_APPEND, append);
    }

    fn truncate(&mut self, truncate: bool) {
        self.flag(libc::O_TRUNC, truncate);
    }

    fn create(&mut self, create: bool) {
        self.flag(libc::O_CREAT, create);
    }

    fn mode(&mut self, mode: raw::mode_t) {
        self.mode = mode as mode_t;
    }
}

impl OpenOptions {
    fn flag(&mut self, bit: c_int, on: bool) {
        if on {
            self.flags |= bit;
        } else {
            self.flags &= !bit;
        }
    }
}

impl sys::File<Fs> for File {
    fn open(path: &OsStr, opts: &OpenOptions) -> Result<File> {
        let path = try!(cstr(path));
        File::open_c(&path, opts)
    }

    fn open_c(path: &CStr, opts: &OpenOptions) -> Result<File> {
        let flags = opts.flags | match (opts.read, opts.write) {
            (true, true) => libc::O_RDWR,
            (false, true) => libc::O_WRONLY,
            (true, false) |
            (false, false) => libc::O_RDONLY,
        };
        let fd = try!(cvt_r(|| unsafe {
            libc::open(path.as_ptr(), flags, opts.mode)
        }));
        let fd = FileDesc::from_inner(fd);
        // Even though we open with the O_CLOEXEC flag, still set CLOEXEC here,
        // in case the open flag is not supported (it's just ignored by the OS
        // in that case).
        fd.set_cloexec();
        Ok(File(fd))
    }

    fn file_attr(&self) -> Result<FileAttr> {
        let mut stat: raw::stat = unsafe { mem::zeroed() };
        if unsafe { libc::fstat(*self.0.as_inner(), &mut stat as *mut _ as *mut _) } < 0 {
            Error::expect_last_result()
        } else {
            Ok(FileAttr { stat: stat })
        }
    }

    fn fsync(&self) -> Result<()> {
        try!(cvt_r(|| unsafe { libc::fsync(*self.0.as_inner()) }));
        Ok(())
    }

    fn datasync(&self) -> Result<()> {
        try!(cvt_r(|| unsafe { os_datasync(*self.0.as_inner()) }));
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

    fn truncate(&self, size: u64) -> Result<()> {
        try!(cvt_r(|| unsafe {
            libc::ftruncate(*self.0.as_inner(), size as libc::off_t)
        }));
        Ok(())
    }
}

impl Read for File {
    fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.0.read(buf)
    }
}

impl Write for File {
    fn write(&self, buf: &[u8]) -> Result<usize> {
        self.0.write(buf)
    }
}

impl Seek for File {
    fn seek(&self, pos: SeekFrom) -> Result<u64> {
        let (whence, pos) = match pos {
            SeekFrom::Start(off) => (libc::SEEK_SET, off as off_t),
            SeekFrom::End(off) => (libc::SEEK_END, off as off_t),
            SeekFrom::Current(off) => (libc::SEEK_CUR, off as off_t),
        };
        match unsafe { libc::lseek(*self.0.as_inner(), pos, whence) } {
            e if e == -1 => Error::expect_last_result(),
            e => Ok(e as u64),
        }
    }
}

impl AsInner<FileDesc> for File {
    fn as_inner(&self) -> &FileDesc { &self.0 }
}

impl IntoInner<FileDesc> for File {
    fn into_inner(self) -> FileDesc { self.0 }
}

impl FromInner<FileDesc> for File {
    fn from_inner(fd: FileDesc) -> Self { File(fd) }
}

impl sys::DirBuilder<Fs> for DirBuilder {
    fn new() -> DirBuilder {
        DirBuilder { mode: 0o777 }
    }

    fn mkdir(&self, p: &OsStr) -> Result<()> {
        let p = try!(cstr(p));
        if unsafe { libc::mkdir(p.as_ptr(), self.mode) } < 0 {
            Error::expect_last_result()
        } else {
            Ok(())
        }
    }

    fn set_mode(&mut self, mode: mode_t) {
        self.mode = mode;
    }
}

fn cstr(path: &OsStr) -> result::Result<CString, NulError> {
    CString::new(path.as_bytes())
}

impl FromInner<c_int> for File {
    fn from_inner(fd: c_int) -> File {
        File(FileDesc::from_inner(fd))
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[cfg(target_os = "linux")]
        fn get_path(fd: c_int) -> Option<OsString> {
            use collections::string::ToString;
            let p = OsString::from_string(format!("/proc/self/fd/{}", fd));
            readlink(&p).ok()
        }

        #[cfg(target_os = "macos")]
        fn get_path(fd: c_int) -> Option<OsString> {
            // FIXME: The use of PATH_MAX is generally not encouraged, but it
            // is inevitable in this case because OS X defines `fcntl` with
            // `F_GETPATH` in terms of `MAXPATHLEN`, and there are no
            // alternatives. If a better method is invented, it should be used
            // instead.
            let mut buf = vec![0;libc::PATH_MAX as usize];
            let n = unsafe { libc::fcntl(fd, libc::F_GETPATH, buf.as_ptr()) };
            if n == -1 {
                return None;
            }
            let l = buf.iter().position(|&c| c == 0).unwrap();
            buf.truncate(l as usize);
            buf.shrink_to_fit();
            Some(OsString::from(OsString::from_vec(buf)))
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        fn get_path(_fd: c_int) -> Option<OsString> {
            // FIXME(#24570): implement this for other Unix platforms
            None
        }

        #[cfg(any(target_os = "linux", target_os = "macos"))]
        fn get_mode(fd: c_int) -> Option<(bool, bool)> {
            let mode = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            if mode == -1 {
                return None;
            }
            match mode & libc::O_ACCMODE {
                libc::O_RDONLY => Some((true, false)),
                libc::O_RDWR => Some((true, true)),
                libc::O_WRONLY => Some((false, true)),
                _ => None
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        fn get_mode(_fd: c_int) -> Option<(bool, bool)> {
            // FIXME(#24570): implement this for other Unix platforms
            None
        }

        let fd = *self.0.as_inner();
        let mut b = f.debug_struct("File");
        b.field("fd", &fd);
        if let Some(path) = get_path(fd) {
            b.field("path", &path);
        }
        if let Some((read, write)) = get_mode(fd) {
            b.field("read", &read).field("write", &write);
        }
        b.finish()
    }
}

fn readdir(p: &OsStr) -> Result<ReadDir> {
    let root = Arc::new(p.to_owned());
    let p = try!(cstr(p));
    unsafe {
        let ptr = libc::opendir(p.as_ptr());
        if ptr.is_null() {
            Error::expect_last_result()
        } else {
            Ok(ReadDir { dirp: Dir(ptr), root: root })
        }
    }
}

fn unlink(p: &OsStr) -> Result<()> {
    let p = try!(cstr(p));
    if unsafe { libc::unlink(p.as_ptr()) } < 0 {
        Error::expect_last_result()
    } else {
        Ok(())
    }
}

fn rename(old: &OsStr, new: &OsStr) -> Result<()> {
    let old = try!(cstr(old));
    let new = try!(cstr(new));
    if unsafe { libc::rename(old.as_ptr(), new.as_ptr()) } < 0 {
        Error::expect_last_result()
    } else {
        Ok(())
    }
}

fn set_perm(p: &OsStr, perm: FilePermissions) -> Result<()> {
    let p = try!(cstr(p));
    cvt_r(|| unsafe { libc::chmod(p.as_ptr(), perm.mode) }).map(drop)
}

fn rmdir(p: &OsStr) -> Result<()> {
    let p = try!(cstr(p));
    if unsafe { libc::rmdir(p.as_ptr()) } < 0 {
        Error::expect_last_result()
    } else {
        Ok(())
    }
}

fn readlink(p: &OsStr) -> Result<OsString> {
    let c_path = try!(cstr(p));
    let p = c_path.as_ptr();

    let mut buf = Vec::with_capacity(256);

    loop {
        let buf_read = match unsafe { libc::readlink(p, buf.as_mut_ptr() as *mut _, buf.capacity() as libc::size_t) } {
            e if e < 0 => return Error::expect_last_result(),
            e => e as usize,
        };

        unsafe { buf.set_len(buf_read); }

        if buf_read != buf.capacity() {
            buf.shrink_to_fit();

            return Ok(OsString::from(OsString::from_vec(buf)));
        }

        // Trigger the internal buffer resizing logic of `Vec` by requiring
        // more space than the current capacity. The length is guaranteed to be
        // the same as the capacity due to the if statement above.
        buf.reserve(1);
    }
}

fn symlink(src: &OsStr, dst: &OsStr) -> Result<()> {
    let src = try!(cstr(src));
    let dst = try!(cstr(dst));
    if unsafe { libc::symlink(src.as_ptr(), dst.as_ptr()) } < 0 {
        Error::expect_last_result()
    } else {
        Ok(())
    }
}

fn link(src: &OsStr, dst: &OsStr) -> Result<()> {
    let src = try!(cstr(src));
    let dst = try!(cstr(dst));
    if unsafe { libc::link(src.as_ptr(), dst.as_ptr()) } < 0 {
        Error::expect_last_result()
    } else {
        Ok(())
    }
}

fn stat(p: &OsStr) -> Result<FileAttr> {
    let p = try!(cstr(p));
    let mut stat: raw::stat = unsafe { mem::zeroed() };
    if unsafe { libc::stat(p.as_ptr(), &mut stat as *mut _ as *mut _) } < 0 {
        Error::expect_last_result()
    } else {
        Ok(FileAttr { stat: stat })
    }
}

fn lstat(p: &OsStr) -> Result<FileAttr> {
    let p = try!(cstr(p));
    let mut stat: raw::stat = unsafe { mem::zeroed() };
    if unsafe { libc::lstat(p.as_ptr(), &mut stat as *mut _ as *mut _) } < 0 {
        Error::expect_last_result()
    } else {
        Ok(FileAttr { stat: stat })
    }
}

fn canonicalize(p: &OsStr) -> Result<OsString> {
    let path = try!(CString::new(p.as_bytes()));
    let buf;
    unsafe {
        let r = c::realpath(path.as_ptr(), ptr::null_mut());
        if r.is_null() {
            return Error::expect_last_result()
        }
        buf = CStr::from_ptr(r).to_bytes().to_vec();
        libc::free(r as *mut _);
    }
    Ok(OsString::from(OsString::from_vec(buf)))
}

pub struct Fs;
impl sys::Fs for Fs {
    type ReadDir = ReadDir;
    type File = File;
    type FileAttr = FileAttr;
    type DirEntry = DirEntry;
    type OpenOptions = OpenOptions;
    type FilePermissions = FilePermissions;
    type FileType = FileType;
    type DirBuilder = DirBuilder;
    type FileHandle = FileDesc;
    type Mode = mode_t;
    type INode = raw::ino_t;

    const COPY_IMP: bool = false;
    fn copy(from: &OsStr, to: &OsStr) -> Result<u64> { unimplemented!() }

    fn unlink(p: &OsStr) -> Result<()> { unlink(p) }
    fn stat(p: &OsStr) -> Result<Self::FileAttr> { stat(p) }
    fn lstat(p: &OsStr) -> Result<Self::FileAttr> { lstat(p) }
    fn rename(from: &OsStr, to: &OsStr) -> Result<()> { rename(from, to) }
    fn link(src: &OsStr, dst: &OsStr) -> Result<()> { link(src, dst) }
    fn symlink(src: &OsStr, dst: &OsStr) -> Result<()> { symlink(src, dst) }
    fn readlink(p: &OsStr) -> Result<OsString> { readlink(p) }
    fn canonicalize(p: &OsStr) -> Result<OsString> { canonicalize(p) }
    fn rmdir(p: &OsStr) -> Result<()> { rmdir(p) }
    fn readdir(p: &OsStr) -> Result<ReadDir> { readdir(p) }
    fn set_perm(p: &OsStr, perm: FilePermissions) -> Result<()> { set_perm(p, perm) }
}
