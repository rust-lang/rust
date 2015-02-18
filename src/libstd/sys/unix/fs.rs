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

use prelude::v1::*;

use ffi::{CString, CStr};
use old_io::{FilePermission, Write, UnstableFileStat, Open, FileAccess, FileMode};
use old_io::{IoResult, FileStat, SeekStyle};
use old_io::{Read, Truncate, SeekCur, SeekSet, ReadWrite, SeekEnd, Append};
use old_io;
use libc::{self, c_int, c_void};
use mem;
use ptr;
use sys::retry;
use sys_common::{keep_going, eof, mkerr_libc};

pub type fd_t = libc::c_int;

pub struct FileDesc {
    /// The underlying C file descriptor.
    fd: fd_t,

    /// Whether to close the file descriptor on drop.
    close_on_drop: bool,
}

impl FileDesc {
    pub fn new(fd: fd_t, close_on_drop: bool) -> FileDesc {
        FileDesc { fd: fd, close_on_drop: close_on_drop }
    }

    pub fn read(&self, buf: &mut [u8]) -> IoResult<uint> {
        let ret = retry(|| unsafe {
            libc::read(self.fd(),
                       buf.as_mut_ptr() as *mut libc::c_void,
                       buf.len() as libc::size_t)
        });
        if ret == 0 {
            Err(eof())
        } else if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(ret as uint)
        }
    }
    pub fn write(&self, buf: &[u8]) -> IoResult<()> {
        let ret = keep_going(buf, |buf, len| {
            unsafe {
                libc::write(self.fd(), buf as *const libc::c_void,
                            len as libc::size_t) as i64
            }
        });
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }

    pub fn fd(&self) -> fd_t { self.fd }

    pub fn seek(&self, pos: i64, whence: SeekStyle) -> IoResult<u64> {
        let whence = match whence {
            SeekSet => libc::SEEK_SET,
            SeekEnd => libc::SEEK_END,
            SeekCur => libc::SEEK_CUR,
        };
        let n = unsafe { libc::lseek(self.fd(), pos as libc::off_t, whence) };
        if n < 0 {
            Err(super::last_error())
        } else {
            Ok(n as u64)
        }
    }

    pub fn tell(&self) -> IoResult<u64> {
        let n = unsafe { libc::lseek(self.fd(), 0, libc::SEEK_CUR) };
        if n < 0 {
            Err(super::last_error())
        } else {
            Ok(n as u64)
        }
    }

    pub fn fsync(&self) -> IoResult<()> {
        mkerr_libc(retry(|| unsafe { libc::fsync(self.fd()) }))
    }

    pub fn datasync(&self) -> IoResult<()> {
        return mkerr_libc(os_datasync(self.fd()));

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        fn os_datasync(fd: c_int) -> c_int {
            unsafe { libc::fcntl(fd, libc::F_FULLFSYNC) }
        }
        #[cfg(target_os = "linux")]
        fn os_datasync(fd: c_int) -> c_int {
            retry(|| unsafe { libc::fdatasync(fd) })
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux")))]
        fn os_datasync(fd: c_int) -> c_int {
            retry(|| unsafe { libc::fsync(fd) })
        }
    }

    pub fn truncate(&self, offset: i64) -> IoResult<()> {
        mkerr_libc(retry(|| unsafe {
            libc::ftruncate(self.fd(), offset as libc::off_t)
        }))
    }

    pub fn fstat(&self) -> IoResult<FileStat> {
        let mut stat: libc::stat = unsafe { mem::zeroed() };
        match unsafe { libc::fstat(self.fd(), &mut stat) } {
            0 => Ok(mkstat(&stat)),
            _ => Err(super::last_error()),
        }
    }

    /// Extract the actual filedescriptor without closing it.
    pub fn unwrap(self) -> fd_t {
        let fd = self.fd;
        unsafe { mem::forget(self) };
        fd
    }
}

impl Drop for FileDesc {
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
                println!("error {} when closing file descriptor {}", n, self.fd);
            }
        }
    }
}

fn cstr(path: &Path) -> IoResult<CString> {
    Ok(try!(CString::new(path.as_vec())))
}

pub fn open(path: &Path, fm: FileMode, fa: FileAccess) -> IoResult<FileDesc> {
    let flags = match fm {
        Open => 0,
        Append => libc::O_APPEND,
        Truncate => libc::O_TRUNC,
    };
    // Opening with a write permission must silently create the file.
    let (flags, mode) = match fa {
        Read => (flags | libc::O_RDONLY, 0),
        Write => (flags | libc::O_WRONLY | libc::O_CREAT,
                        libc::S_IRUSR | libc::S_IWUSR),
        ReadWrite => (flags | libc::O_RDWR | libc::O_CREAT,
                            libc::S_IRUSR | libc::S_IWUSR),
    };

    let path = try!(cstr(path));
    match retry(|| unsafe { libc::open(path.as_ptr(), flags, mode) }) {
        -1 => Err(super::last_error()),
        fd => Ok(FileDesc::new(fd, true)),
    }
}

pub fn mkdir(p: &Path, mode: uint) -> IoResult<()> {
    let p = try!(cstr(p));
    mkerr_libc(unsafe { libc::mkdir(p.as_ptr(), mode as libc::mode_t) })
}

pub fn readdir(p: &Path) -> IoResult<Vec<Path>> {
    use libc::{dirent_t};
    use libc::{opendir, readdir_r, closedir};

    fn prune(root: &CString, dirs: Vec<Path>) -> Vec<Path> {
        let root = Path::new(root);

        dirs.into_iter().filter(|path| {
            path.as_vec() != b"." && path.as_vec() != b".."
        }).map(|path| root.join(path)).collect()
    }

    extern {
        fn rust_dirent_t_size() -> libc::c_int;
        fn rust_list_dir_val(ptr: *mut dirent_t) -> *const libc::c_char;
    }

    let size = unsafe { rust_dirent_t_size() };
    let mut buf = Vec::<u8>::with_capacity(size as uint);
    let ptr = buf.as_mut_ptr() as *mut dirent_t;

    let p = try!(CString::new(p.as_vec()));
    let dir_ptr = unsafe {opendir(p.as_ptr())};

    if dir_ptr as uint != 0 {
        let mut paths = vec!();
        let mut entry_ptr = ptr::null_mut();
        while unsafe { readdir_r(dir_ptr, ptr, &mut entry_ptr) == 0 } {
            if entry_ptr.is_null() { break }
            paths.push(unsafe {
                Path::new(CStr::from_ptr(rust_list_dir_val(entry_ptr)).to_bytes())
            });
        }
        assert_eq!(unsafe { closedir(dir_ptr) }, 0);
        Ok(prune(&p, paths))
    } else {
        Err(super::last_error())
    }
}

pub fn unlink(p: &Path) -> IoResult<()> {
    let p = try!(cstr(p));
    mkerr_libc(unsafe { libc::unlink(p.as_ptr()) })
}

pub fn rename(old: &Path, new: &Path) -> IoResult<()> {
    let old = try!(cstr(old));
    let new = try!(cstr(new));
    mkerr_libc(unsafe {
        libc::rename(old.as_ptr(), new.as_ptr())
    })
}

pub fn chmod(p: &Path, mode: uint) -> IoResult<()> {
    let p = try!(cstr(p));
    mkerr_libc(retry(|| unsafe {
        libc::chmod(p.as_ptr(), mode as libc::mode_t)
    }))
}

pub fn rmdir(p: &Path) -> IoResult<()> {
    let p = try!(cstr(p));
    mkerr_libc(unsafe { libc::rmdir(p.as_ptr()) })
}

pub fn chown(p: &Path, uid: int, gid: int) -> IoResult<()> {
    let p = try!(cstr(p));
    mkerr_libc(retry(|| unsafe {
        libc::chown(p.as_ptr(), uid as libc::uid_t, gid as libc::gid_t)
    }))
}

pub fn readlink(p: &Path) -> IoResult<Path> {
    let c_path = try!(cstr(p));
    let p = c_path.as_ptr();
    let mut len = unsafe { libc::pathconf(p as *mut _, libc::_PC_NAME_MAX) };
    if len == -1 {
        len = 1024; // FIXME: read PATH_MAX from C ffi?
    }
    let mut buf: Vec<u8> = Vec::with_capacity(len as uint);
    match unsafe {
        libc::readlink(p, buf.as_ptr() as *mut libc::c_char,
                       len as libc::size_t) as libc::c_int
    } {
        -1 => Err(super::last_error()),
        n => {
            assert!(n > 0);
            unsafe { buf.set_len(n as uint); }
            Ok(Path::new(buf))
        }
    }
}

pub fn symlink(src: &Path, dst: &Path) -> IoResult<()> {
    let src = try!(cstr(src));
    let dst = try!(cstr(dst));
    mkerr_libc(unsafe { libc::symlink(src.as_ptr(), dst.as_ptr()) })
}

pub fn link(src: &Path, dst: &Path) -> IoResult<()> {
    let src = try!(cstr(src));
    let dst = try!(cstr(dst));
    mkerr_libc(unsafe { libc::link(src.as_ptr(), dst.as_ptr()) })
}

fn mkstat(stat: &libc::stat) -> FileStat {
    // FileStat times are in milliseconds
    fn mktime(secs: u64, nsecs: u64) -> u64 { secs * 1000 + nsecs / 1000000 }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn flags(stat: &libc::stat) -> u64 { stat.st_flags as u64 }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn flags(_stat: &libc::stat) -> u64 { 0 }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn gen(stat: &libc::stat) -> u64 { stat.st_gen as u64 }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn gen(_stat: &libc::stat) -> u64 { 0 }

    FileStat {
        size: stat.st_size as u64,
        kind: match (stat.st_mode as libc::mode_t) & libc::S_IFMT {
            libc::S_IFREG => old_io::FileType::RegularFile,
            libc::S_IFDIR => old_io::FileType::Directory,
            libc::S_IFIFO => old_io::FileType::NamedPipe,
            libc::S_IFBLK => old_io::FileType::BlockSpecial,
            libc::S_IFLNK => old_io::FileType::Symlink,
            _ => old_io::FileType::Unknown,
        },
        perm: FilePermission::from_bits_truncate(stat.st_mode as u32),
        created: mktime(stat.st_ctime as u64, stat.st_ctime_nsec as u64),
        modified: mktime(stat.st_mtime as u64, stat.st_mtime_nsec as u64),
        accessed: mktime(stat.st_atime as u64, stat.st_atime_nsec as u64),
        unstable: UnstableFileStat {
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
        },
    }
}

pub fn stat(p: &Path) -> IoResult<FileStat> {
    let p = try!(cstr(p));
    let mut stat: libc::stat = unsafe { mem::zeroed() };
    match unsafe { libc::stat(p.as_ptr(), &mut stat) } {
        0 => Ok(mkstat(&stat)),
        _ => Err(super::last_error()),
    }
}

pub fn lstat(p: &Path) -> IoResult<FileStat> {
    let p = try!(cstr(p));
    let mut stat: libc::stat = unsafe { mem::zeroed() };
    match unsafe { libc::lstat(p.as_ptr(), &mut stat) } {
        0 => Ok(mkstat(&stat)),
        _ => Err(super::last_error()),
    }
}

pub fn utime(p: &Path, atime: u64, mtime: u64) -> IoResult<()> {
    let p = try!(cstr(p));
    let buf = libc::utimbuf {
        actime: (atime / 1000) as libc::time_t,
        modtime: (mtime / 1000) as libc::time_t,
    };
    mkerr_libc(unsafe { libc::utime(p.as_ptr(), &buf) })
}

#[cfg(test)]
mod tests {
    use super::FileDesc;
    use libc;
    use os;
    use prelude::v1::*;

    #[cfg_attr(any(target_os = "freebsd",
                   target_os = "openbsd"),
               ignore)]
    // under some system, pipe(2) will return a bidrectionnal pipe
    #[test]
    fn test_file_desc() {
        // Run this test with some pipes so we don't have to mess around with
        // opening or closing files.
        let os::Pipe { reader, writer } = unsafe { os::pipe().unwrap() };
        let mut reader = FileDesc::new(reader, true);
        let mut writer = FileDesc::new(writer, true);

        writer.write(b"test").ok().unwrap();
        let mut buf = [0u8; 4];
        match reader.read(&mut buf) {
            Ok(4) => {
                assert_eq!(buf[0], 't' as u8);
                assert_eq!(buf[1], 'e' as u8);
                assert_eq!(buf[2], 's' as u8);
                assert_eq!(buf[3], 't' as u8);
            }
            r => panic!("invalid read: {:?}", r),
        }

        assert!(writer.read(&mut buf).is_err());
        assert!(reader.write(&buf).is_err());
    }
}
