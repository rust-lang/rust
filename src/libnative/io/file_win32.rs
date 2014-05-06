// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Blocking win32-based file I/O

use std::c_str::CString;
use std::cast;
use std::io::IoError;
use std::io;
use libc::{c_int, c_void};
use libc;
use std::mem;
use std::os::win32::{as_utf16_p, fill_utf16_buf_and_decode};
use std::ptr;
use std::rt::rtio;
use std::str;
use std::sync::arc::UnsafeArc;
use std::slice;

use io::IoResult;

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

    pub fn inner_read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        let mut read = 0;
        let ret = unsafe {
            libc::ReadFile(self.handle(), buf.as_ptr() as libc::LPVOID,
                           buf.len() as libc::DWORD, &mut read,
                           ptr::mut_null())
        };
        if ret != 0 {
            Ok(read as uint)
        } else {
            Err(super::last_error())
        }
    }
    pub fn inner_write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let mut cur = buf.as_ptr();
        let mut remaining = buf.len();
        while remaining > 0 {
            let mut amt = 0;
            let ret = unsafe {
                libc::WriteFile(self.handle(), cur as libc::LPVOID,
                                remaining as libc::DWORD, &mut amt,
                                ptr::mut_null())
            };
            if ret != 0 {
                remaining -= amt as uint;
                cur = unsafe { cur.offset(amt as int) };
            } else {
                return Err(super::last_error())
            }
        }
        Ok(())
    }

    pub fn fd(&self) -> fd_t {
        // This unsafety is fine because we're just reading off the file
        // descriptor, no one is modifying this.
        unsafe { (*self.inner.get()).fd }
    }

    pub fn handle(&self) -> libc::HANDLE {
        unsafe { libc::get_osfhandle(self.fd()) as libc::HANDLE }
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
        let mut read = 0;
        let mut overlap: libc::OVERLAPPED = unsafe { mem::init() };
        overlap.Offset = offset as libc::DWORD;
        overlap.OffsetHigh = (offset >> 32) as libc::DWORD;
        let ret = unsafe {
            libc::ReadFile(self.handle(), buf.as_ptr() as libc::LPVOID,
                           buf.len() as libc::DWORD, &mut read,
                           &mut overlap)
        };
        if ret != 0 {
            Ok(read as int)
        } else {
            Err(super::last_error())
        }
    }
    fn pwrite(&mut self, buf: &[u8], mut offset: u64) -> Result<(), IoError> {
        let mut cur = buf.as_ptr();
        let mut remaining = buf.len();
        let mut overlap: libc::OVERLAPPED = unsafe { mem::init() };
        while remaining > 0 {
            overlap.Offset = offset as libc::DWORD;
            overlap.OffsetHigh = (offset >> 32) as libc::DWORD;
            let mut amt = 0;
            let ret = unsafe {
                libc::WriteFile(self.handle(), cur as libc::LPVOID,
                                remaining as libc::DWORD, &mut amt,
                                &mut overlap)
            };
            if ret != 0 {
                remaining -= amt as uint;
                cur = unsafe { cur.offset(amt as int) };
                offset += amt as u64;
            } else {
                return Err(super::last_error())
            }
        }
        Ok(())
    }
    fn seek(&mut self, pos: i64, style: io::SeekStyle) -> Result<u64, IoError> {
        let whence = match style {
            io::SeekSet => libc::FILE_BEGIN,
            io::SeekEnd => libc::FILE_END,
            io::SeekCur => libc::FILE_CURRENT,
        };
        unsafe {
            let mut newpos = 0;
            match libc::SetFilePointerEx(self.handle(), pos, &mut newpos,
                                         whence) {
                0 => Err(super::last_error()),
                _ => Ok(newpos as u64),
            }
        }
    }
    fn tell(&self) -> Result<u64, IoError> {
        // This transmute is fine because our seek implementation doesn't
        // actually use the mutable self at all.
        // FIXME #13933: Remove/justify all `&T` to `&mut T` transmutes
        unsafe { cast::transmute::<&_, &mut FileDesc>(self).seek(0, io::SeekCur) }
    }

    fn fsync(&mut self) -> Result<(), IoError> {
        super::mkerr_winbool(unsafe {
            libc::FlushFileBuffers(self.handle())
        })
    }

    fn datasync(&mut self) -> Result<(), IoError> { return self.fsync(); }

    fn truncate(&mut self, offset: i64) -> Result<(), IoError> {
        let orig_pos = try!(self.tell());
        let _ = try!(self.seek(offset, io::SeekSet));
        let ret = unsafe {
            match libc::SetEndOfFile(self.handle()) {
                0 => Err(super::last_error()),
                _ => Ok(())
            }
        };
        let _ = self.seek(orig_pos as i64, io::SeekSet);
        return ret;
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
                println!("error {} when closing file descriptor {}", n, self.fd);
            }
        }
    }
}

pub fn open(path: &CString, fm: io::FileMode, fa: io::FileAccess)
        -> IoResult<FileDesc> {
    // Flags passed to open_osfhandle
    let flags = match fm {
        io::Open => 0,
        io::Append => libc::O_APPEND,
        io::Truncate => libc::O_TRUNC,
    };
    let flags = match fa {
        io::Read => flags | libc::O_RDONLY,
        io::Write => flags | libc::O_WRONLY | libc::O_CREAT,
        io::ReadWrite => flags | libc::O_RDWR | libc::O_CREAT,
    };

    let mut dwDesiredAccess = match fa {
        io::Read => libc::FILE_GENERIC_READ,
        io::Write => libc::FILE_GENERIC_WRITE,
        io::ReadWrite => libc::FILE_GENERIC_READ | libc::FILE_GENERIC_WRITE
    };

    // libuv has a good comment about this, but the basic idea is what we try to
    // emulate unix semantics by enabling all sharing by allowing things such as
    // deleting a file while it's still open.
    let dwShareMode = libc::FILE_SHARE_READ | libc::FILE_SHARE_WRITE |
                      libc::FILE_SHARE_DELETE;

    let dwCreationDisposition = match (fm, fa) {
        (io::Truncate, io::Read) => libc::TRUNCATE_EXISTING,
        (io::Truncate, _) => libc::CREATE_ALWAYS,
        (io::Open, io::Read) => libc::OPEN_EXISTING,
        (io::Open, _) => libc::OPEN_ALWAYS,
        (io::Append, io::Read) => {
            dwDesiredAccess |= libc::FILE_APPEND_DATA;
            libc::OPEN_EXISTING
        }
        (io::Append, _) => {
            dwDesiredAccess &= !libc::FILE_WRITE_DATA;
            dwDesiredAccess |= libc::FILE_APPEND_DATA;
            libc::OPEN_ALWAYS
        }
    };

    let mut dwFlagsAndAttributes = libc::FILE_ATTRIBUTE_NORMAL;
    // Compat with unix, this allows opening directories (see libuv)
    dwFlagsAndAttributes |= libc::FILE_FLAG_BACKUP_SEMANTICS;

    let handle = as_utf16_p(path.as_str().unwrap(), |buf| unsafe {
        libc::CreateFileW(buf,
                          dwDesiredAccess,
                          dwShareMode,
                          ptr::mut_null(),
                          dwCreationDisposition,
                          dwFlagsAndAttributes,
                          ptr::mut_null())
    });
    if handle == libc::INVALID_HANDLE_VALUE as libc::HANDLE {
        Err(super::last_error())
    } else {
        let fd = unsafe {
            libc::open_osfhandle(handle as libc::intptr_t, flags)
        };
        if fd < 0 {
            let _ = unsafe { libc::CloseHandle(handle) };
            Err(super::last_error())
        } else {
            Ok(FileDesc::new(fd, true))
        }
    }
}

pub fn mkdir(p: &CString, _mode: io::FilePermission) -> IoResult<()> {
    super::mkerr_winbool(unsafe {
        // FIXME: turn mode into something useful? #2623
        as_utf16_p(p.as_str().unwrap(), |buf| {
            libc::CreateDirectoryW(buf, ptr::mut_null())
        })
    })
}

pub fn readdir(p: &CString) -> IoResult<Vec<Path>> {
    use std::rt::global_heap::malloc_raw;

    fn prune(root: &CString, dirs: Vec<Path>) -> Vec<Path> {
        let root = unsafe { CString::new(root.with_ref(|p| p), false) };
        let root = Path::new(root);

        dirs.move_iter().filter(|path| {
            path.as_vec() != bytes!(".") && path.as_vec() != bytes!("..")
        }).map(|path| root.join(path)).collect()
    }

    extern {
        fn rust_list_dir_wfd_size() -> libc::size_t;
        fn rust_list_dir_wfd_fp_buf(wfd: *libc::c_void) -> *u16;
    }
    let star = Path::new(unsafe {
        CString::new(p.with_ref(|p| p), false)
    }).join("*");
    as_utf16_p(star.as_str().unwrap(), |path_ptr| unsafe {
        let wfd_ptr = malloc_raw(rust_list_dir_wfd_size() as uint);
        let find_handle = libc::FindFirstFileW(path_ptr, wfd_ptr as libc::HANDLE);
        if find_handle as libc::c_int != libc::INVALID_HANDLE_VALUE {
            let mut paths = vec!();
            let mut more_files = 1 as libc::c_int;
            while more_files != 0 {
                let fp_buf = rust_list_dir_wfd_fp_buf(wfd_ptr as *c_void);
                if fp_buf as uint == 0 {
                    fail!("os::list_dir() failure: got null ptr from wfd");
                } else {
                    let fp_vec = slice::from_buf(fp_buf, libc::wcslen(fp_buf) as uint);
                    let fp_trimmed = str::truncate_utf16_at_nul(fp_vec);
                    let fp_str = str::from_utf16(fp_trimmed)
                            .expect("rust_list_dir_wfd_fp_buf returned invalid UTF-16");
                    paths.push(Path::new(fp_str));
                }
                more_files = libc::FindNextFileW(find_handle,
                                                 wfd_ptr as libc::HANDLE);
            }
            assert!(libc::FindClose(find_handle) != 0);
            libc::free(wfd_ptr as *mut c_void);
            Ok(prune(p, paths))
        } else {
            Err(super::last_error())
        }
    })
}

pub fn unlink(p: &CString) -> IoResult<()> {
    super::mkerr_winbool(unsafe {
        as_utf16_p(p.as_str().unwrap(), |buf| {
            libc::DeleteFileW(buf)
        })
    })
}

pub fn rename(old: &CString, new: &CString) -> IoResult<()> {
    super::mkerr_winbool(unsafe {
        as_utf16_p(old.as_str().unwrap(), |old| {
            as_utf16_p(new.as_str().unwrap(), |new| {
                libc::MoveFileExW(old, new, libc::MOVEFILE_REPLACE_EXISTING)
            })
        })
    })
}

pub fn chmod(p: &CString, mode: io::FilePermission) -> IoResult<()> {
    super::mkerr_libc(as_utf16_p(p.as_str().unwrap(), |p| unsafe {
        libc::wchmod(p, mode.bits() as libc::c_int)
    }))
}

pub fn rmdir(p: &CString) -> IoResult<()> {
    super::mkerr_libc(as_utf16_p(p.as_str().unwrap(), |p| unsafe {
        libc::wrmdir(p)
    }))
}

pub fn chown(_p: &CString, _uid: int, _gid: int) -> IoResult<()> {
    // libuv has this as a no-op, so seems like this should as well?
    Ok(())
}

pub fn readlink(p: &CString) -> IoResult<Path> {
    // FIXME: I have a feeling that this reads intermediate symlinks as well.
    let handle = unsafe {
        as_utf16_p(p.as_str().unwrap(), |p| {
            libc::CreateFileW(p,
                              libc::GENERIC_READ,
                              libc::FILE_SHARE_READ,
                              ptr::mut_null(),
                              libc::OPEN_EXISTING,
                              libc::FILE_ATTRIBUTE_NORMAL,
                              ptr::mut_null())
        })
    };
    if handle as int == libc::INVALID_HANDLE_VALUE as int {
        return Err(super::last_error())
    }
    // Specify (sz - 1) because the documentation states that it's the size
    // without the null pointer
    let ret = fill_utf16_buf_and_decode(|buf, sz| unsafe {
        libc::GetFinalPathNameByHandleW(handle,
                                        buf as *u16,
                                        sz - 1,
                                        libc::VOLUME_NAME_DOS)
    });
    let ret = match ret {
        Some(ref s) if s.starts_with(r"\\?\") => Ok(Path::new(s.slice_from(4))),
        Some(s) => Ok(Path::new(s)),
        None => Err(super::last_error()),
    };
    assert!(unsafe { libc::CloseHandle(handle) } != 0);
    return ret;
}

pub fn symlink(src: &CString, dst: &CString) -> IoResult<()> {
    super::mkerr_winbool(as_utf16_p(src.as_str().unwrap(), |src| {
        as_utf16_p(dst.as_str().unwrap(), |dst| {
            unsafe { libc::CreateSymbolicLinkW(dst, src, 0) }
        }) as libc::BOOL
    }))
}

pub fn link(src: &CString, dst: &CString) -> IoResult<()> {
    super::mkerr_winbool(as_utf16_p(src.as_str().unwrap(), |src| {
        as_utf16_p(dst.as_str().unwrap(), |dst| {
            unsafe { libc::CreateHardLinkW(dst, src, ptr::mut_null()) }
        })
    }))
}

fn mkstat(stat: &libc::stat, path: &CString) -> io::FileStat {
    let path = unsafe { CString::new(path.with_ref(|p| p), false) };
    let kind = match (stat.st_mode as c_int) & libc::S_IFMT {
        libc::S_IFREG => io::TypeFile,
        libc::S_IFDIR => io::TypeDirectory,
        libc::S_IFIFO => io::TypeNamedPipe,
        libc::S_IFBLK => io::TypeBlockSpecial,
        libc::S_IFLNK => io::TypeSymlink,
        _ => io::TypeUnknown,
    };

    io::FileStat {
        path: Path::new(path),
        size: stat.st_size as u64,
        kind: kind,
        perm: unsafe {
          io::FilePermission::from_bits(stat.st_mode as u32)  & io::AllPermissions
        },
        created: stat.st_ctime as u64,
        modified: stat.st_mtime as u64,
        accessed: stat.st_atime as u64,
        unstable: io::UnstableFileStat {
            device: stat.st_dev as u64,
            inode: stat.st_ino as u64,
            rdev: stat.st_rdev as u64,
            nlink: stat.st_nlink as u64,
            uid: stat.st_uid as u64,
            gid: stat.st_gid as u64,
            blksize: 0,
            blocks: 0,
            flags: 0,
            gen: 0,
        }
    }
}

pub fn stat(p: &CString) -> IoResult<io::FileStat> {
    let mut stat: libc::stat = unsafe { mem::uninit() };
    as_utf16_p(p.as_str().unwrap(), |up| {
        match unsafe { libc::wstat(up, &mut stat) } {
            0 => Ok(mkstat(&stat, p)),
            _ => Err(super::last_error()),
        }
    })
}

pub fn lstat(_p: &CString) -> IoResult<io::FileStat> {
    // FIXME: implementation is missing
    Err(super::unimpl())
}

pub fn utime(p: &CString, atime: u64, mtime: u64) -> IoResult<()> {
    let buf = libc::utimbuf {
        actime: (atime / 1000) as libc::time64_t,
        modtime: (mtime / 1000) as libc::time64_t,
    };
    super::mkerr_libc(as_utf16_p(p.as_str().unwrap(), |p| unsafe {
        libc::wutime(p, &buf)
    }))
}
