// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::error::{self, Result};
use sys::inner::*;
use ffi::{OsStr, OsString};
use io;

use borrow::Cow;
use ffi::{CString, CStr, NulError};
use os::unix::ffi::{OsStrExt, OsStringExt};
use libc::{self, c_int, size_t, off_t, c_char, mode_t};
use fmt;
use mem;
use result;
use ptr;
use sync::Arc;
use sys::unix::fd::FileDesc;
use sys::unix::platform::raw;
use sys::unix::{c, cvt, cvt_r};
use vec::Vec;
use borrow::ToOwned;

pub struct File(FileDesc);
impl_inner!(File(FileDesc));
impl_inner!(File(FileDesc(c_int)));

#[derive(Clone)]
pub struct FileAttr(raw::stat);
impl_inner!(FileAttr(raw::stat));

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
pub struct FilePermissions(mode_t);
impl_inner!(FilePermissions(raw::mode_t));

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct FileType(mode_t);
impl_inner!(FileType(mode_t));

pub struct DirBuilder { mode: mode_t }

impl FileAttr {
    pub fn size(&self) -> u64 { self.0.st_size as u64 }
    pub fn perm(&self) -> FilePermissions {
        FilePermissions((self.0.st_mode as mode_t) & 0o777)
    }

    pub fn file_type(&self) -> FileType {
        FileType(self.0.st_mode as mode_t)
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool { self.0 & 0o222 == 0 }
    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.0 &= !0o222;
        } else {
            self.0 |= 0o222;
        }
    }
    pub fn mode(&self) -> raw::mode_t { self.0 }
}

impl FileType {
    pub fn is_dir(&self) -> bool { self.is(libc::S_IFDIR) }
    pub fn is_file(&self) -> bool { self.is(libc::S_IFREG) }
    pub fn is_symlink(&self) -> bool { self.is(libc::S_IFLNK) }
}

impl FileType {
    fn is(&self, mode: mode_t) -> bool { self.0 & libc::S_IFMT == mode }

    pub fn is_block_device(&self) -> bool { self.is(libc::S_IFBLK) }
    pub fn is_char_device(&self) -> bool { self.is(libc::S_IFCHR) }
    pub fn is_fifo(&self) -> bool { self.is(libc::S_IFIFO) }
    pub fn is_socket(&self) -> bool { self.is(libc::S_IFSOCK) }
}

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
                return Some(error::expect_last_result())
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

impl DirEntry {
    pub fn file_name(&self) -> Cow<OsStr> {
        Cow::Borrowed(OsStr::from_bytes(self.name_bytes()))
    }

    pub fn root(&self) -> &OsStr {
        &self.root
    }

    pub fn metadata(&self) -> Result<FileAttr> {
        lstat(&self.path())
    }

    pub fn file_type(&self) -> Result<FileType> {
        extern {
            fn rust_dir_get_mode(ptr: *mut libc::dirent_t) -> c_int;
        }
        unsafe {
            match rust_dir_get_mode(self.dirent()) {
                -1 => lstat(&self.path()).map(|a| a.file_type()),
                n => Ok(FileType(n as mode_t)),
            }
        }
    }

    pub fn ino(&self) -> raw::ino_t {
        extern {
            fn rust_dir_get_ino(ptr: *mut libc::dirent_t) -> raw::ino_t;
        }
        unsafe { rust_dir_get_ino(self.dirent()) }
    }
}

impl DirEntry {
    pub fn name_bytes(&self) -> &[u8] {
        extern {
            fn rust_list_dir_val(ptr: *mut libc::dirent_t) -> *const c_char;
        }
        unsafe {
            CStr::from_ptr(rust_list_dir_val(self.dirent())).to_bytes()
        }
    }

    pub fn dirent(&self) -> *mut libc::dirent_t {
        self.buf.as_ptr() as *mut _
    }

    fn path(&self) -> OsString {
        use path::PathBuf;

        let mut path = PathBuf::from((*self.root).clone());
        path.push(&*self.file_name());
        path.into()
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            flags: libc::O_CLOEXEC,
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

    pub fn mode(&mut self, mode: raw::mode_t) {
        self.mode = mode as mode_t;
    }
}

impl OpenOptions {
    pub fn flag(&mut self, bit: c_int, on: bool) {
        if on {
            self.flags |= bit;
        } else {
            self.flags &= !bit;
        }
    }
}

impl File {
    pub fn open(path: &OsStr, opts: &OpenOptions) -> Result<File> {
        let path = try!(cstr(path));
        File::open_c(&path, opts)
    }

    pub fn open_c(path: &CStr, opts: &OpenOptions) -> Result<File> {
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

    pub fn file_attr(&self) -> Result<FileAttr> {
        let mut stat: raw::stat = unsafe { mem::zeroed() };
        if unsafe { libc::fstat(*self.0.as_inner(), &mut stat as *mut _ as *mut _) } < 0 {
            error::expect_last_result()
        } else {
            Ok(FileAttr(stat))
        }
    }

    pub fn fsync(&self) -> Result<()> {
        try!(cvt_r(|| unsafe { libc::fsync(*self.0.as_inner()) }));
        Ok(())
    }

    pub fn datasync(&self) -> Result<()> {
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

    pub fn truncate(&self, size: u64) -> Result<()> {
        try!(cvt_r(|| unsafe {
            libc::ftruncate(*self.0.as_inner(), size as libc::off_t)
        }));
        Ok(())
    }

    pub fn read(&self, buf: &mut [u8]) -> Result<usize> {
        self.0.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> Result<usize> {
        self.0.write(buf)
    }

    pub fn flush(&self) -> Result<()> { Ok(()) }

    pub fn seek(&self, pos: io::SeekFrom) -> Result<u64> {
        self.0.seek(pos)
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder { mode: 0o777 }
    }

    pub fn mkdir(&self, p: &OsStr) -> Result<()> {
        let p = try!(cstr(p));
        cvt(unsafe { libc::mkdir(p.as_ptr(), self.mode) }).map(drop)
    }

    pub fn set_mode(&mut self, mode: mode_t) {
        self.mode = mode;
    }
}

fn cstr(path: &OsStr) -> result::Result<CString, NulError> {
    CString::new(path.as_bytes())
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[cfg(target_os = "linux")]
        fn get_path(fd: c_int) -> Option<OsString> {
            let p = OsString::from_vec(format!("/proc/self/fd/{}", fd).into());
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

pub fn readdir(p: &OsStr) -> Result<ReadDir> {
    let root = Arc::new(p.to_owned());
    let p = try!(cstr(p));
    unsafe {
        let ptr = libc::opendir(p.as_ptr());
        if ptr.is_null() {
            error::expect_last_result()
        } else {
            Ok(ReadDir { dirp: Dir(ptr), root: root })
        }
    }
}

pub fn unlink(p: &OsStr) -> Result<()> {
    let p = try!(cstr(p));
    cvt(unsafe { libc::unlink(p.as_ptr()) }).map(drop)
}

pub fn rename(old: &OsStr, new: &OsStr) -> Result<()> {
    let old = try!(cstr(old));
    let new = try!(cstr(new));
    cvt(unsafe { libc::rename(old.as_ptr(), new.as_ptr()) }).map(drop)
}

pub fn set_perm(p: &OsStr, perm: FilePermissions) -> Result<()> {
    let p = try!(cstr(p));
    cvt_r(|| unsafe { libc::chmod(p.as_ptr(), *perm.as_inner()) }).map(drop)
}

pub fn rmdir(p: &OsStr) -> Result<()> {
    let p = try!(cstr(p));
    cvt(unsafe { libc::rmdir(p.as_ptr()) }).map(drop)
}

pub fn readlink(p: &OsStr) -> Result<OsString> {
    let c_path = try!(cstr(p));
    let p = c_path.as_ptr();

    let mut buf = Vec::with_capacity(256);

    loop {
        let buf_read = try!(cvt(unsafe { libc::readlink(p, buf.as_mut_ptr() as *mut _, buf.capacity() as libc::size_t) })) as usize;

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

pub fn symlink(src: &OsStr, dst: &OsStr) -> Result<()> {
    let src = try!(cstr(src));
    let dst = try!(cstr(dst));
    cvt(unsafe { libc::symlink(src.as_ptr(), dst.as_ptr()) }).map(drop)
}

pub fn link(src: &OsStr, dst: &OsStr) -> Result<()> {
    let src = try!(cstr(src));
    let dst = try!(cstr(dst));
    cvt(unsafe { libc::link(src.as_ptr(), dst.as_ptr()) }).map(drop)
}

pub fn stat(p: &OsStr) -> Result<FileAttr> {
    let p = try!(cstr(p));
    let mut stat: raw::stat = unsafe { mem::zeroed() };
    try!(cvt(unsafe { libc::stat(p.as_ptr(), &mut stat as *mut _ as *mut _) }));
    Ok(FileAttr(stat))
}

pub fn lstat(p: &OsStr) -> Result<FileAttr> {
    let p = try!(cstr(p));
    let mut stat: raw::stat = unsafe { mem::zeroed() };
    try!(cvt(unsafe { libc::lstat(p.as_ptr(), &mut stat as *mut _ as *mut _) }));
    Ok(FileAttr(stat))
}

pub fn canonicalize(p: &OsStr) -> Result<OsString> {
    let path = try!(CString::new(p.as_bytes()));
    let buf;
    unsafe {
        let r = c::realpath(path.as_ptr(), ptr::null_mut());
        if r.is_null() {
            return error::expect_last_result()
        }
        buf = CStr::from_ptr(r).to_bytes().to_vec();
        libc::free(r as *mut _);
    }
    Ok(OsString::from_vec(buf))
}

pub const COPY_IMP: bool = false;
#[inline(always)]
pub fn copy(_from: &OsStr, _to: &OsStr) -> Result<u64> { unimplemented!() }

pub type FileHandle = FileDesc;
