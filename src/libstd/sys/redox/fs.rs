// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use os::unix::prelude::*;

use ffi::{CString, CStr, OsString, OsStr};
use fmt;
use io::{self, Error, ErrorKind, SeekFrom};
use libc::{self, c_int, mode_t};
use mem;
use path::{Path, PathBuf};
use sync::Arc;
use sys::fd::FileDesc;
use sys::time::SystemTime;
use sys::{cvt, cvt_r};
use sys_common::{AsInner, FromInner};

use libc::{stat as stat64, fstat as fstat64, lstat as lstat64, off_t as off64_t,
           ftruncate as ftruncate64, lseek as lseek64, open as open64};

pub struct File(FileDesc);

#[derive(Clone)]
pub struct FileAttr {
    stat: stat64,
}

pub struct ReadDir {
    data: Vec<u8>,
    i: usize,
    root: Arc<PathBuf>,
}

struct Dir(FileDesc);

unsafe impl Send for Dir {}
unsafe impl Sync for Dir {}

pub struct DirEntry {
    root: Arc<PathBuf>,
    name: Box<[u8]>
}

#[derive(Clone)]
pub struct OpenOptions {
    // generic
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    // system-specific
    custom_flags: i32,
    mode: mode_t,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions { mode: mode_t }

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileType { mode: mode_t }

pub struct DirBuilder { mode: mode_t }

impl FileAttr {
    pub fn size(&self) -> u64 { self.stat.st_size as u64 }
    pub fn perm(&self) -> FilePermissions {
        FilePermissions { mode: (self.stat.st_mode as mode_t) & 0o777 }
    }

    pub fn file_type(&self) -> FileType {
        FileType { mode: self.stat.st_mode as mode_t }
    }
}

impl FileAttr {
    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_mtime as libc::time_t,
            tv_nsec: self.stat.st_mtime_nsec as i32,
        }))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_atime as libc::time_t,
            tv_nsec: self.stat.st_atime_nsec as i32,
        }))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(libc::timespec {
            tv_sec: self.stat.st_ctime as libc::time_t,
            tv_nsec: self.stat.st_ctime_nsec as i32,
        }))
    }
}

impl AsInner<stat64> for FileAttr {
    fn as_inner(&self) -> &stat64 { &self.stat }
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
    pub fn mode(&self) -> u32 { self.mode as u32 }
}

impl FileType {
    pub fn is_dir(&self) -> bool { self.is(libc::MODE_DIR) }
    pub fn is_file(&self) -> bool { self.is(libc::MODE_FILE) }
    pub fn is_symlink(&self) -> bool { false }

    pub fn is(&self, mode: mode_t) -> bool { self.mode & (libc::MODE_DIR | libc::MODE_FILE) == mode }
}

impl FromInner<u32> for FilePermissions {
    fn from_inner(mode: u32) -> FilePermissions {
        FilePermissions { mode: mode as mode_t }
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // This will only be called from std::fs::ReadDir, which will add a "ReadDir()" frame.
        // Thus the result will be e g 'ReadDir("/home")'
        fmt::Debug::fmt(&*self.root, f)
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        loop {
            let start = self.i;
            while self.i < self.data.len() {
                let i = self.i;
                self.i += 1;
                if self.data[i] == b'\n' {
                    break;
                }
            }
            if start < self.i {
                let ret = DirEntry {
                    name: self.data[start .. self.i].to_owned().into_boxed_slice(),
                    root: self.root.clone()
                };
                if ret.name_bytes() != b"." && ret.name_bytes() != b".." {
                    return Some(Ok(ret))
                }
            } else {
                return None;
            }
        }
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.root.join(OsStr::from_bytes(self.name_bytes()))
    }

    pub fn file_name(&self) -> OsString {
        OsStr::from_bytes(self.name_bytes()).to_os_string()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        lstat(&self.path())
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        stat(&self.path()).map(|m| m.file_type())
    }

    fn name_bytes(&self) -> &[u8] {
        &*self.name
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            // generic
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            // system-specific
            custom_flags: 0,
            mode: 0o666,
        }
    }

    pub fn read(&mut self, read: bool) { self.read = read; }
    pub fn write(&mut self, write: bool) { self.write = write; }
    pub fn append(&mut self, append: bool) { self.append = append; }
    pub fn truncate(&mut self, truncate: bool) { self.truncate = truncate; }
    pub fn create(&mut self, create: bool) { self.create = create; }
    pub fn create_new(&mut self, create_new: bool) { self.create_new = create_new; }

    pub fn custom_flags(&mut self, flags: i32) { self.custom_flags = flags; }
    pub fn mode(&mut self, mode: u32) { self.mode = mode as mode_t; }

    fn get_access_mode(&self) -> io::Result<c_int> {
        match (self.read, self.write, self.append) {
            (true,  false, false) => Ok(libc::O_RDONLY as c_int),
            (false, true,  false) => Ok(libc::O_WRONLY as c_int),
            (true,  true,  false) => Ok(libc::O_RDWR as c_int),
            (false, _,     true)  => Ok(libc::O_WRONLY as c_int | libc::O_APPEND as c_int),
            (true,  _,     true)  => Ok(libc::O_RDWR as c_int | libc::O_APPEND as c_int),
            (false, false, false) => Err(Error::from_raw_os_error(libc::EINVAL)),
        }
    }

    fn get_creation_mode(&self) -> io::Result<c_int> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) =>
                if self.truncate || self.create || self.create_new {
                    return Err(Error::from_raw_os_error(libc::EINVAL));
                },
            (_, true) =>
                if self.truncate && !self.create_new {
                    return Err(Error::from_raw_os_error(libc::EINVAL));
                },
        }

        Ok(match (self.create, self.truncate, self.create_new) {
                (false, false, false) => 0,
                (true,  false, false) => libc::O_CREAT as c_int,
                (false, true,  false) => libc::O_TRUNC as c_int,
                (true,  true,  false) => libc::O_CREAT as c_int | libc::O_TRUNC as c_int,
                (_,      _,    true)  => libc::O_CREAT as c_int | libc::O_EXCL as c_int,
           })
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path = cstr(path)?;
        File::open_c(&path, opts)
    }

    pub fn open_c(path: &CStr, opts: &OpenOptions) -> io::Result<File> {
        let flags = libc::O_CLOEXEC as i32 |
                    opts.get_access_mode()? |
                    opts.get_creation_mode()? |
                    (opts.custom_flags as usize & !libc::O_ACCMODE) as i32;
        let fd = cvt_r(|| unsafe {
            open64(path.as_ptr(), flags, opts.mode as mode_t)
        })?;
        let fd = FileDesc::new(fd);

        Ok(File(fd))
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let mut stat: stat64 = unsafe { mem::zeroed() };
        cvt(unsafe {
            fstat64(self.0.raw(), &mut stat)
        })?;
        Ok(FileAttr { stat: stat })
    }

    pub fn fsync(&self) -> io::Result<()> {
        cvt_r(|| unsafe { libc::fsync(self.0.raw()) })?;
        Ok(())
    }

    pub fn datasync(&self) -> io::Result<()> {
        cvt_r(|| unsafe { os_datasync(self.0.raw()) })?;
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
        #[cfg(target_os = "android")]
        return ::sys::android::ftruncate64(self.0.raw(), size);

        #[cfg(not(target_os = "android"))]
        return cvt_r(|| unsafe {
            ftruncate64(self.0.raw(), size as off64_t)
        }).map(|_| ());
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn flush(&self) -> io::Result<()> { Ok(()) }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            // Casting to `i64` is fine, too large values will end up as
            // negative which will cause an error in `lseek64`.
            SeekFrom::Start(off) => (libc::SEEK_SET, off as i64),
            SeekFrom::End(off) => (libc::SEEK_END, off),
            SeekFrom::Current(off) => (libc::SEEK_CUR, off),
        };
        let n = cvt(unsafe { lseek64(self.0.raw(), pos as usize, whence) } as isize)?;
        Ok(n as u64)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        self.0.duplicate().map(File)
    }

    pub fn dup(&self, buf: &[u8]) -> io::Result<File> {
        libc::dup_extra(*self.fd().as_inner() as usize, buf)
            .map(|fd| File(FileDesc::new(fd as i32)))
            .map_err(|err| Error::from_raw_os_error(err.errno))
    }

    pub fn path(&self) -> io::Result<PathBuf> {
        let mut buf: [u8; 4096] = [0; 4096];
        match libc::fpath(*self.fd().as_inner() as usize, &mut buf) {
            Ok(count) => Ok(PathBuf::from(unsafe { String::from_utf8_unchecked(Vec::from(&buf[0..count])) })),
            Err(err) => Err(Error::from_raw_os_error(err.errno)),
        }
    }

    pub fn fd(&self) -> &FileDesc { &self.0 }

    pub fn into_fd(self) -> FileDesc { self.0 }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder { mode: 0o777 }
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        let p = cstr(p)?;
        cvt(unsafe { libc::mkdir(p.as_ptr(), self.mode) })?;
        Ok(())
    }

    pub fn set_mode(&mut self, mode: u32) {
        self.mode = mode as mode_t;
    }
}

fn cstr(path: &Path) -> io::Result<CString> {
    Ok(CString::new(path.as_os_str().as_bytes())?)
}

impl FromInner<c_int> for File {
    fn from_inner(fd: c_int) -> File {
        File(FileDesc::new(fd))
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        #[cfg(target_os = "linux")]
        fn get_path(fd: c_int) -> Option<PathBuf> {
            let mut p = PathBuf::from("/proc/self/fd");
            p.push(&fd.to_string());
            readlink(&p).ok()
        }

        #[cfg(target_os = "macos")]
        fn get_path(fd: c_int) -> Option<PathBuf> {
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
            Some(PathBuf::from(OsString::from_vec(buf)))
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        fn get_path(_fd: c_int) -> Option<PathBuf> {
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

        let fd = self.0.raw();
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

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let root = Arc::new(p.to_path_buf());
    let p = cstr(p)?;
    unsafe {
        let fd = FileDesc::new(cvt(libc::open(p.as_ptr(), 0, 0))?);
        let mut data = Vec::new();
        fd.read_to_end(&mut data)?;
        Ok(ReadDir { data: data, i: 0, root: root })
    }
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let p = cstr(p)?;
    cvt(unsafe { libc::unlink(p.as_ptr()) })?;
    Ok(())
}

pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> {
    unimplemented!();
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    unimplemented!();
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let p = cstr(p)?;
    cvt(unsafe { libc::rmdir(p.as_ptr()) })?;
    Ok(())
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let filetype = lstat(path)?.file_type();
    if filetype.is_symlink() {
        unlink(path)
    } else {
        remove_dir_all_recursive(path)
    }
}

fn remove_dir_all_recursive(path: &Path) -> io::Result<()> {
    for child in readdir(path)? {
        let child = child?;
        if child.file_type()?.is_dir() {
            remove_dir_all_recursive(&child.path())?;
        } else {
            unlink(&child.path())?;
        }
    }
    rmdir(path)
}

pub fn readlink(_p: &Path) -> io::Result<PathBuf> {
    unimplemented!();
}

pub fn symlink(_src: &Path, _dst: &Path) -> io::Result<()> {
    unimplemented!();
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    unimplemented!();
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let p = cstr(p)?;
    let mut stat: stat64 = unsafe { mem::zeroed() };
    cvt(unsafe {
        stat64(p.as_ptr(), &mut stat as *mut _ as *mut _)
    })?;
    Ok(FileAttr { stat: stat })
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    let p = cstr(p)?;
    let mut stat: stat64 = unsafe { mem::zeroed() };
    cvt(unsafe {
        lstat64(p.as_ptr(), &mut stat as *mut _ as *mut _)
    })?;
    Ok(FileAttr { stat: stat })
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unimplemented!();
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    use fs::{File, set_permissions};
    if !from.is_file() {
        return Err(Error::new(ErrorKind::InvalidInput,
                              "the source path is not an existing regular file"))
    }

    let mut reader = File::open(from)?;
    let mut writer = File::create(to)?;
    let perm = reader.metadata()?.permissions();

    let ret = io::copy(&mut reader, &mut writer)?;
    set_permissions(to, perm)?;
    Ok(ret)
}
