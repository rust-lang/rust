// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Blocking posix-based file I/O

use std::sync::arc::UnsafeArc;
use std::c_str::CString;
use std::io::IoError;
use std::io;
use libc::{c_int, c_void};
use libc;
use std::mem;
use std::rt::rtio;

use io::{IoResult, retry, keep_going};

pub type fd_t = libc::c_int;

struct Inner {
    fd: fd_t,
    close_on_drop: bool,
}

pub struct FileDesc {
    inner: UnsafeArc<Inner>
}

impl FileDesc {
    /// Create a `FileDesc` from an open C file descriptor.
    ///
    /// The `FileDesc` will take ownership of the specified file descriptor and
    /// close it upon destruction if the `close_on_drop` flag is true, otherwise
    /// it will not close the file descriptor when this `FileDesc` is dropped.
    ///
    /// Note that all I/O operations done on this object will be *blocking*, but
    /// they do not require the runtime to be active.
    pub fn new(fd: fd_t, close_on_drop: bool) -> FileDesc {
        FileDesc { inner: UnsafeArc::new(Inner {
            fd: fd,
            close_on_drop: close_on_drop
        }) }
    }

    // FIXME(#10465) these functions should not be public, but anything in
    //               native::io wanting to use them is forced to have all the
    //               rtio traits in scope
    pub fn inner_read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        let ret = retry(|| unsafe {
            libc::read(self.fd(),
                       buf.as_mut_ptr() as *mut libc::c_void,
                       buf.len() as libc::size_t) as libc::c_int
        });
        if ret == 0 {
            Err(io::standard_error(io::EndOfFile))
        } else if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(ret as uint)
        }
    }
    pub fn inner_write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let ret = keep_going(buf, |buf, len| {
            unsafe {
                libc::write(self.fd(), buf as *libc::c_void,
                            len as libc::size_t) as i64
            }
        });
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }

    pub fn fd(&self) -> fd_t {
        // This unsafety is fine because we're just reading off the file
        // descriptor, no one is modifying this.
        unsafe { (*self.inner.get()).fd }
    }
}

impl io::Reader for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
        self.inner_read(buf)
    }
}

impl io::Writer for FileDesc {
    fn write(&mut self, buf: &[u8]) -> io::IoResult<()> {
        self.inner_write(buf)
    }
}

impl rtio::RtioFileStream for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> Result<int, IoError> {
        self.inner_read(buf).map(|i| i as int)
    }
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        self.inner_write(buf)
    }
    fn pread(&mut self, buf: &mut [u8], offset: u64) -> Result<int, IoError> {
        match retry(|| unsafe {
            libc::pread(self.fd(), buf.as_ptr() as *libc::c_void,
                        buf.len() as libc::size_t,
                        offset as libc::off_t) as libc::c_int
        }) {
            -1 => Err(super::last_error()),
            n => Ok(n as int)
        }
    }
    fn pwrite(&mut self, buf: &[u8], offset: u64) -> Result<(), IoError> {
        super::mkerr_libc(retry(|| unsafe {
            libc::pwrite(self.fd(), buf.as_ptr() as *libc::c_void,
                         buf.len() as libc::size_t, offset as libc::off_t)
        } as c_int))
    }
    fn seek(&mut self, pos: i64, whence: io::SeekStyle) -> Result<u64, IoError> {
        let whence = match whence {
            io::SeekSet => libc::SEEK_SET,
            io::SeekEnd => libc::SEEK_END,
            io::SeekCur => libc::SEEK_CUR,
        };
        let n = unsafe { libc::lseek(self.fd(), pos as libc::off_t, whence) };
        if n < 0 {
            Err(super::last_error())
        } else {
            Ok(n as u64)
        }
    }
    fn tell(&self) -> Result<u64, IoError> {
        let n = unsafe { libc::lseek(self.fd(), 0, libc::SEEK_CUR) };
        if n < 0 {
            Err(super::last_error())
        } else {
            Ok(n as u64)
        }
    }
    fn fsync(&mut self) -> Result<(), IoError> {
        super::mkerr_libc(retry(|| unsafe { libc::fsync(self.fd()) }))
    }
    fn datasync(&mut self) -> Result<(), IoError> {
        return super::mkerr_libc(os_datasync(self.fd()));

        #[cfg(target_os = "macos")]
        fn os_datasync(fd: c_int) -> c_int {
            unsafe { libc::fcntl(fd, libc::F_FULLFSYNC) }
        }
        #[cfg(target_os = "linux")]
        fn os_datasync(fd: c_int) -> c_int {
            retry(|| unsafe { libc::fdatasync(fd) })
        }
        #[cfg(not(target_os = "macos"), not(target_os = "linux"))]
        fn os_datasync(fd: c_int) -> c_int {
            retry(|| unsafe { libc::fsync(fd) })
        }
    }
    fn truncate(&mut self, offset: i64) -> Result<(), IoError> {
        super::mkerr_libc(retry(|| unsafe {
            libc::ftruncate(self.fd(), offset as libc::off_t)
        }))
    }
}

impl rtio::RtioPipe for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        self.inner_read(buf)
    }
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        self.inner_write(buf)
    }
    fn clone(&self) -> ~rtio::RtioPipe:Send {
        box FileDesc { inner: self.inner.clone() } as ~rtio::RtioPipe:Send
    }
}

impl rtio::RtioTTY for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        self.inner_read(buf)
    }
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        self.inner_write(buf)
    }
    fn set_raw(&mut self, _raw: bool) -> Result<(), IoError> {
        Err(super::unimpl())
    }
    fn get_winsize(&mut self) -> Result<(int, int), IoError> {
        Err(super::unimpl())
    }
    fn isatty(&self) -> bool { false }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // closing stdio file handles makes no sense, so never do it. Also, note
        // that errors are ignored when closing a file descriptor. The reason
        // for this is that if an error occurs we don't actually know if the
        // file descriptor was closed or not, and if we retried (for something
        // like EINTR), we might close another valid file descriptor (opened
        // after we closed ours.
        if self.close_on_drop && self.fd > libc::STDERR_FILENO {
            let n = unsafe { libc::close(self.fd) };
            if n != 0 {
                println!("error {} when closing file descriptor {}", n,
                         self.fd);
            }
        }
    }
}

pub struct CFile {
    file: *libc::FILE,
    fd: FileDesc,
}

impl CFile {
    /// Create a `CFile` from an open `FILE` pointer.
    ///
    /// The `CFile` takes ownership of the `FILE` pointer and will close it upon
    /// destruction.
    pub fn new(file: *libc::FILE) -> CFile {
        CFile {
            file: file,
            fd: FileDesc::new(unsafe { libc::fileno(file) }, false)
        }
    }

    pub fn flush(&mut self) -> Result<(), IoError> {
        super::mkerr_libc(retry(|| unsafe { libc::fflush(self.file) }))
    }
}

impl rtio::RtioFileStream for CFile {
    fn read(&mut self, buf: &mut [u8]) -> Result<int, IoError> {
        let ret = keep_going(buf, |buf, len| {
            unsafe {
                libc::fread(buf as *mut libc::c_void, 1, len as libc::size_t,
                            self.file) as i64
            }
        });
        if ret == 0 {
            Err(io::standard_error(io::EndOfFile))
        } else if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(ret as int)
        }
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let ret = keep_going(buf, |buf, len| {
            unsafe {
                libc::fwrite(buf as *libc::c_void, 1, len as libc::size_t,
                            self.file) as i64
            }
        });
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }

    fn pread(&mut self, buf: &mut [u8], offset: u64) -> Result<int, IoError> {
        self.flush().and_then(|()| self.fd.pread(buf, offset))
    }
    fn pwrite(&mut self, buf: &[u8], offset: u64) -> Result<(), IoError> {
        self.flush().and_then(|()| self.fd.pwrite(buf, offset))
    }
    fn seek(&mut self, pos: i64, style: io::SeekStyle) -> Result<u64, IoError> {
        let whence = match style {
            io::SeekSet => libc::SEEK_SET,
            io::SeekEnd => libc::SEEK_END,
            io::SeekCur => libc::SEEK_CUR,
        };
        let n = unsafe { libc::fseek(self.file, pos as libc::c_long, whence) };
        if n < 0 {
            Err(super::last_error())
        } else {
            Ok(n as u64)
        }
    }
    fn tell(&self) -> Result<u64, IoError> {
        let ret = unsafe { libc::ftell(self.file) };
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(ret as u64)
        }
    }
    fn fsync(&mut self) -> Result<(), IoError> {
        self.flush().and_then(|()| self.fd.fsync())
    }
    fn datasync(&mut self) -> Result<(), IoError> {
        self.flush().and_then(|()| self.fd.fsync())
    }
    fn truncate(&mut self, offset: i64) -> Result<(), IoError> {
        self.flush().and_then(|()| self.fd.truncate(offset))
    }
}

impl Drop for CFile {
    fn drop(&mut self) {
        unsafe { let _ = libc::fclose(self.file); }
    }
}

pub fn open(path: &CString, fm: io::FileMode, fa: io::FileAccess)
        -> IoResult<FileDesc> {
    let flags = match fm {
        io::Open => 0,
        io::Append => libc::O_APPEND,
        io::Truncate => libc::O_TRUNC,
    };
    // Opening with a write permission must silently create the file.
    let (flags, mode) = match fa {
        io::Read => (flags | libc::O_RDONLY, 0),
        io::Write => (flags | libc::O_WRONLY | libc::O_CREAT,
                      libc::S_IRUSR | libc::S_IWUSR),
        io::ReadWrite => (flags | libc::O_RDWR | libc::O_CREAT,
                          libc::S_IRUSR | libc::S_IWUSR),
    };

    match retry(|| unsafe { libc::open(path.with_ref(|p| p), flags, mode) }) {
        -1 => Err(super::last_error()),
        fd => Ok(FileDesc::new(fd, true)),
    }
}

pub fn mkdir(p: &CString, mode: io::FilePermission) -> IoResult<()> {
    super::mkerr_libc(retry(|| unsafe {
        libc::mkdir(p.with_ref(|p| p), mode.bits() as libc::mode_t)
    }))
}

pub fn readdir(p: &CString) -> IoResult<Vec<Path>> {
    use libc::{dirent_t};
    use libc::{opendir, readdir_r, closedir};

    fn prune(root: &CString, dirs: Vec<Path>) -> Vec<Path> {
        let root = unsafe { CString::new(root.with_ref(|p| p), false) };
        let root = Path::new(root);

        dirs.move_iter().filter(|path| {
            path.as_vec() != bytes!(".") && path.as_vec() != bytes!("..")
        }).map(|path| root.join(path)).collect()
    }

    extern {
        fn rust_dirent_t_size() -> libc::c_int;
        fn rust_list_dir_val(ptr: *mut dirent_t) -> *libc::c_char;
    }

    let size = unsafe { rust_dirent_t_size() };
    let mut buf = Vec::<u8>::with_capacity(size as uint);
    let ptr = buf.as_mut_slice().as_mut_ptr() as *mut dirent_t;

    let dir_ptr = p.with_ref(|buf| unsafe { opendir(buf) });

    if dir_ptr as uint != 0 {
        let mut paths = vec!();
        let mut entry_ptr = 0 as *mut dirent_t;
        while unsafe { readdir_r(dir_ptr, ptr, &mut entry_ptr) == 0 } {
            if entry_ptr.is_null() { break }
            let cstr = unsafe {
                CString::new(rust_list_dir_val(entry_ptr), false)
            };
            paths.push(Path::new(cstr));
        }
        assert_eq!(unsafe { closedir(dir_ptr) }, 0);
        Ok(prune(p, paths))
    } else {
        Err(super::last_error())
    }
}

pub fn unlink(p: &CString) -> IoResult<()> {
    super::mkerr_libc(retry(|| unsafe { libc::unlink(p.with_ref(|p| p)) }))
}

pub fn rename(old: &CString, new: &CString) -> IoResult<()> {
    super::mkerr_libc(retry(|| unsafe {
        libc::rename(old.with_ref(|p| p), new.with_ref(|p| p))
    }))
}

pub fn chmod(p: &CString, mode: io::FilePermission) -> IoResult<()> {
    super::mkerr_libc(retry(|| unsafe {
        libc::chmod(p.with_ref(|p| p), mode.bits() as libc::mode_t)
    }))
}

pub fn rmdir(p: &CString) -> IoResult<()> {
    super::mkerr_libc(retry(|| unsafe {
        libc::rmdir(p.with_ref(|p| p))
    }))
}

pub fn chown(p: &CString, uid: int, gid: int) -> IoResult<()> {
    super::mkerr_libc(retry(|| unsafe {
        libc::chown(p.with_ref(|p| p), uid as libc::uid_t,
                    gid as libc::gid_t)
    }))
}

pub fn readlink(p: &CString) -> IoResult<Path> {
    let p = p.with_ref(|p| p);
    let mut len = unsafe { libc::pathconf(p, libc::_PC_NAME_MAX) };
    if len == -1 {
        len = 1024; // FIXME: read PATH_MAX from C ffi?
    }
    let mut buf: Vec<u8> = Vec::with_capacity(len as uint);
    match retry(|| unsafe {
        libc::readlink(p, buf.as_ptr() as *mut libc::c_char,
                       len as libc::size_t) as libc::c_int
    }) {
        -1 => Err(super::last_error()),
        n => {
            assert!(n > 0);
            unsafe { buf.set_len(n as uint); }
            Ok(Path::new(buf))
        }
    }
}

pub fn symlink(src: &CString, dst: &CString) -> IoResult<()> {
    super::mkerr_libc(retry(|| unsafe {
        libc::symlink(src.with_ref(|p| p), dst.with_ref(|p| p))
    }))
}

pub fn link(src: &CString, dst: &CString) -> IoResult<()> {
    super::mkerr_libc(retry(|| unsafe {
        libc::link(src.with_ref(|p| p), dst.with_ref(|p| p))
    }))
}

fn mkstat(stat: &libc::stat, path: &CString) -> io::FileStat {
    let path = unsafe { CString::new(path.with_ref(|p| p), false) };

    // FileStat times are in milliseconds
    fn mktime(secs: u64, nsecs: u64) -> u64 { secs * 1000 + nsecs / 1000000 }

    let kind = match (stat.st_mode as c_int) & libc::S_IFMT {
        libc::S_IFREG => io::TypeFile,
        libc::S_IFDIR => io::TypeDirectory,
        libc::S_IFIFO => io::TypeNamedPipe,
        libc::S_IFBLK => io::TypeBlockSpecial,
        libc::S_IFLNK => io::TypeSymlink,
        _ => io::TypeUnknown,
    };

    #[cfg(not(target_os = "linux"), not(target_os = "android"))]
    fn flags(stat: &libc::stat) -> u64 { stat.st_flags as u64 }
    #[cfg(target_os = "linux")] #[cfg(target_os = "android")]
    fn flags(_stat: &libc::stat) -> u64 { 0 }

    #[cfg(not(target_os = "linux"), not(target_os = "android"))]
    fn gen(stat: &libc::stat) -> u64 { stat.st_gen as u64 }
    #[cfg(target_os = "linux")] #[cfg(target_os = "android")]
    fn gen(_stat: &libc::stat) -> u64 { 0 }

    io::FileStat {
        path: Path::new(path),
        size: stat.st_size as u64,
        kind: kind,
        perm: unsafe {
            io::FilePermission::from_bits(stat.st_mode as u32) & io::AllPermissions
        },
        created: mktime(stat.st_ctime as u64, stat.st_ctime_nsec as u64),
        modified: mktime(stat.st_mtime as u64, stat.st_mtime_nsec as u64),
        accessed: mktime(stat.st_atime as u64, stat.st_atime_nsec as u64),
        unstable: io::UnstableFileStat {
            device: stat.st_dev as u64,
            inode: stat.st_ino as u64,
            rdev: stat.st_rdev as u64,
            nlink: stat.st_nlink as u64,
            uid: stat.st_uid as u64,
            gid: stat.st_gid as u64,
            blksize: stat.st_blksize as u64,
            blocks: stat.st_blocks as u64,
            flags: flags(stat),
            gen: gen(stat),
        }
    }
}

pub fn stat(p: &CString) -> IoResult<io::FileStat> {
    let mut stat: libc::stat = unsafe { mem::uninit() };
    match retry(|| unsafe { libc::stat(p.with_ref(|p| p), &mut stat) }) {
        0 => Ok(mkstat(&stat, p)),
        _ => Err(super::last_error()),
    }
}

pub fn lstat(p: &CString) -> IoResult<io::FileStat> {
    let mut stat: libc::stat = unsafe { mem::uninit() };
    match retry(|| unsafe { libc::lstat(p.with_ref(|p| p), &mut stat) }) {
        0 => Ok(mkstat(&stat, p)),
        _ => Err(super::last_error()),
    }
}

pub fn utime(p: &CString, atime: u64, mtime: u64) -> IoResult<()> {
    let buf = libc::utimbuf {
        actime: (atime / 1000) as libc::time_t,
        modtime: (mtime / 1000) as libc::time_t,
    };
    super::mkerr_libc(retry(|| unsafe {
        libc::utime(p.with_ref(|p| p), &buf)
    }))
}

#[cfg(test)]
mod tests {
    use super::{CFile, FileDesc};
    use std::io;
    use libc;
    use std::os;
    use std::rt::rtio::RtioFileStream;

    #[ignore(cfg(target_os = "freebsd"))] // hmm, maybe pipes have a tiny buffer
    #[test]
    fn test_file_desc() {
        // Run this test with some pipes so we don't have to mess around with
        // opening or closing files.
        let os::Pipe { input, out } = os::pipe();
        let mut reader = FileDesc::new(input, true);
        let mut writer = FileDesc::new(out, true);

        writer.inner_write(bytes!("test")).unwrap();
        let mut buf = [0u8, ..4];
        match reader.inner_read(buf) {
            Ok(4) => {
                assert_eq!(buf[0], 't' as u8);
                assert_eq!(buf[1], 'e' as u8);
                assert_eq!(buf[2], 's' as u8);
                assert_eq!(buf[3], 't' as u8);
            }
            r => fail!("invalid read: {:?}", r)
        }

        assert!(writer.inner_read(buf).is_err());
        assert!(reader.inner_write(buf).is_err());
    }

    #[test]
    fn test_cfile() {
        unsafe {
            let f = libc::tmpfile();
            assert!(!f.is_null());
            let mut file = CFile::new(f);

            file.write(bytes!("test")).unwrap();
            let mut buf = [0u8, ..4];
            let _ = file.seek(0, io::SeekSet).unwrap();
            match file.read(buf) {
                Ok(4) => {
                    assert_eq!(buf[0], 't' as u8);
                    assert_eq!(buf[1], 'e' as u8);
                    assert_eq!(buf[2], 's' as u8);
                    assert_eq!(buf[3], 't' as u8);
                }
                r => fail!("invalid read: {:?}", r)
            }
        }
    }
}
