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

use alloc::arc::Arc;
use libc::{c_int, c_void};
use libc;
use std::c_str::CString;
use std::mem;
use std::os::win32::fill_utf16_buf_and_decode;
use std::ptr;
use std::rt::rtio;
use std::rt::rtio::{IoResult, IoError};
use std::str;
use std::vec;

pub type fd_t = libc::c_int;

struct Inner {
    fd: fd_t,
    close_on_drop: bool,
}

pub struct FileDesc {
    inner: Arc<Inner>
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
        FileDesc { inner: Arc::new(Inner {
            fd: fd,
            close_on_drop: close_on_drop
        }) }
    }

    pub fn inner_read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
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
    pub fn inner_write(&mut self, buf: &[u8]) -> IoResult<()> {
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

    pub fn fd(&self) -> fd_t { self.inner.fd }

    pub fn handle(&self) -> libc::HANDLE {
        unsafe { libc::get_osfhandle(self.fd()) as libc::HANDLE }
    }

    // A version of seek that takes &self so that tell can call it
    //   - the private seek should of course take &mut self.
    fn seek_common(&self, pos: i64, style: rtio::SeekStyle) -> IoResult<u64> {
        let whence = match style {
            rtio::SeekSet => libc::FILE_BEGIN,
            rtio::SeekEnd => libc::FILE_END,
            rtio::SeekCur => libc::FILE_CURRENT,
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

}

impl rtio::RtioFileStream for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<int> {
        self.inner_read(buf).map(|i| i as int)
    }
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.inner_write(buf)
    }

    fn pread(&mut self, buf: &mut [u8], offset: u64) -> IoResult<int> {
        let mut read = 0;
        let mut overlap: libc::OVERLAPPED = unsafe { mem::zeroed() };
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
    fn pwrite(&mut self, buf: &[u8], mut offset: u64) -> IoResult<()> {
        let mut cur = buf.as_ptr();
        let mut remaining = buf.len();
        let mut overlap: libc::OVERLAPPED = unsafe { mem::zeroed() };
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

    fn seek(&mut self, pos: i64, style: rtio::SeekStyle) -> IoResult<u64> {
        self.seek_common(pos, style)
    }

    fn tell(&self) -> IoResult<u64> {
        self.seek_common(0, rtio::SeekCur)
    }

    fn fsync(&mut self) -> IoResult<()> {
        super::mkerr_winbool(unsafe {
            libc::FlushFileBuffers(self.handle())
        })
    }

    fn datasync(&mut self) -> IoResult<()> { return self.fsync(); }

    fn truncate(&mut self, offset: i64) -> IoResult<()> {
        let orig_pos = try!(self.tell());
        let _ = try!(self.seek(offset, rtio::SeekSet));
        let ret = unsafe {
            match libc::SetEndOfFile(self.handle()) {
                0 => Err(super::last_error()),
                _ => Ok(())
            }
        };
        let _ = self.seek(orig_pos as i64, rtio::SeekSet);
        return ret;
    }

    fn fstat(&mut self) -> IoResult<rtio::FileStat> {
        let mut stat: libc::stat = unsafe { mem::zeroed() };
        match unsafe { libc::fstat(self.fd(), &mut stat) } {
            0 => Ok(mkstat(&stat)),
            _ => Err(super::last_error()),
        }
    }
}

impl rtio::RtioPipe for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.inner_read(buf)
    }
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.inner_write(buf)
    }
    fn clone(&self) -> Box<rtio::RtioPipe + Send> {
        box FileDesc { inner: self.inner.clone() } as Box<rtio::RtioPipe + Send>
    }

    // Only supported on named pipes currently. Note that this doesn't have an
    // impact on the std::io primitives, this is never called via
    // std::io::PipeStream. If the functionality is exposed in the future, then
    // these methods will need to be implemented.
    fn close_read(&mut self) -> IoResult<()> {
        Err(super::unimpl())
    }
    fn close_write(&mut self) -> IoResult<()> {
        Err(super::unimpl())
    }
    fn set_timeout(&mut self, _t: Option<u64>) {}
    fn set_read_timeout(&mut self, _t: Option<u64>) {}
    fn set_write_timeout(&mut self, _t: Option<u64>) {}
}

impl rtio::RtioTTY for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.inner_read(buf)
    }
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.inner_write(buf)
    }
    fn set_raw(&mut self, _raw: bool) -> IoResult<()> {
        Err(super::unimpl())
    }
    fn get_winsize(&mut self) -> IoResult<(int, int)> {
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

pub fn to_utf16(s: &CString) -> IoResult<Vec<u16>> {
    match s.as_str() {
        Some(s) => Ok(s.to_utf16().append_one(0)),
        None => Err(IoError {
            code: libc::ERROR_INVALID_NAME as uint,
            extra: 0,
            detail: Some("valid unicode input required".to_str()),
        })
    }
}

pub fn open(path: &CString, fm: rtio::FileMode, fa: rtio::FileAccess)
        -> IoResult<FileDesc> {
    // Flags passed to open_osfhandle
    let flags = match fm {
        rtio::Open => 0,
        rtio::Append => libc::O_APPEND,
        rtio::Truncate => libc::O_TRUNC,
    };
    let flags = match fa {
        rtio::Read => flags | libc::O_RDONLY,
        rtio::Write => flags | libc::O_WRONLY | libc::O_CREAT,
        rtio::ReadWrite => flags | libc::O_RDWR | libc::O_CREAT,
    };

    let mut dwDesiredAccess = match fa {
        rtio::Read => libc::FILE_GENERIC_READ,
        rtio::Write => libc::FILE_GENERIC_WRITE,
        rtio::ReadWrite => libc::FILE_GENERIC_READ | libc::FILE_GENERIC_WRITE
    };

    // libuv has a good comment about this, but the basic idea is what we try to
    // emulate unix semantics by enabling all sharing by allowing things such as
    // deleting a file while it's still open.
    let dwShareMode = libc::FILE_SHARE_READ | libc::FILE_SHARE_WRITE |
                      libc::FILE_SHARE_DELETE;

    let dwCreationDisposition = match (fm, fa) {
        (rtio::Truncate, rtio::Read) => libc::TRUNCATE_EXISTING,
        (rtio::Truncate, _) => libc::CREATE_ALWAYS,
        (rtio::Open, rtio::Read) => libc::OPEN_EXISTING,
        (rtio::Open, _) => libc::OPEN_ALWAYS,
        (rtio::Append, rtio::Read) => {
            dwDesiredAccess |= libc::FILE_APPEND_DATA;
            libc::OPEN_EXISTING
        }
        (rtio::Append, _) => {
            dwDesiredAccess &= !libc::FILE_WRITE_DATA;
            dwDesiredAccess |= libc::FILE_APPEND_DATA;
            libc::OPEN_ALWAYS
        }
    };

    let mut dwFlagsAndAttributes = libc::FILE_ATTRIBUTE_NORMAL;
    // Compat with unix, this allows opening directories (see libuv)
    dwFlagsAndAttributes |= libc::FILE_FLAG_BACKUP_SEMANTICS;

    let path = try!(to_utf16(path));
    let handle = unsafe {
        libc::CreateFileW(path.as_ptr(),
                          dwDesiredAccess,
                          dwShareMode,
                          ptr::mut_null(),
                          dwCreationDisposition,
                          dwFlagsAndAttributes,
                          ptr::mut_null())
    };
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

pub fn mkdir(p: &CString, _mode: uint) -> IoResult<()> {
    let p = try!(to_utf16(p));
    super::mkerr_winbool(unsafe {
        // FIXME: turn mode into something useful? #2623
        libc::CreateDirectoryW(p.as_ptr(), ptr::mut_null())
    })
}

pub fn readdir(p: &CString) -> IoResult<Vec<CString>> {
    use std::rt::libc_heap::malloc_raw;

    fn prune(root: &CString, dirs: Vec<Path>) -> Vec<CString> {
        let root = unsafe { CString::new(root.with_ref(|p| p), false) };
        let root = Path::new(root);

        dirs.move_iter().filter(|path| {
            path.as_vec() != bytes!(".") && path.as_vec() != bytes!("..")
        }).map(|path| root.join(path).to_c_str()).collect()
    }

    extern {
        fn rust_list_dir_wfd_size() -> libc::size_t;
        fn rust_list_dir_wfd_fp_buf(wfd: *libc::c_void) -> *u16;
    }
    let star = Path::new(unsafe {
        CString::new(p.with_ref(|p| p), false)
    }).join("*");
    let path = try!(to_utf16(&star.to_c_str()));

    unsafe {
        let wfd_ptr = malloc_raw(rust_list_dir_wfd_size() as uint);
        let find_handle = libc::FindFirstFileW(path.as_ptr(), wfd_ptr as libc::HANDLE);
        if find_handle as libc::c_int != libc::INVALID_HANDLE_VALUE {
            let mut paths = vec!();
            let mut more_files = 1 as libc::c_int;
            while more_files != 0 {
                let fp_buf = rust_list_dir_wfd_fp_buf(wfd_ptr as *c_void);
                if fp_buf as uint == 0 {
                    fail!("os::list_dir() failure: got null ptr from wfd");
                } else {
                    let fp_vec = vec::raw::from_buf(fp_buf, libc::wcslen(fp_buf) as uint);
                    let fp_trimmed = str::truncate_utf16_at_nul(fp_vec.as_slice());
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
    }
}

pub fn unlink(p: &CString) -> IoResult<()> {
    let p = try!(to_utf16(p));
    super::mkerr_winbool(unsafe {
        libc::DeleteFileW(p.as_ptr())
    })
}

pub fn rename(old: &CString, new: &CString) -> IoResult<()> {
    let old = try!(to_utf16(old));
    let new = try!(to_utf16(new));
    super::mkerr_winbool(unsafe {
        libc::MoveFileExW(old.as_ptr(), new.as_ptr(),
                          libc::MOVEFILE_REPLACE_EXISTING)
    })
}

pub fn chmod(p: &CString, mode: uint) -> IoResult<()> {
    let p = try!(to_utf16(p));
    super::mkerr_libc(unsafe {
        libc::wchmod(p.as_ptr(), mode as libc::c_int)
    })
}

pub fn rmdir(p: &CString) -> IoResult<()> {
    let p = try!(to_utf16(p));
    super::mkerr_libc(unsafe { libc::wrmdir(p.as_ptr()) })
}

pub fn chown(_p: &CString, _uid: int, _gid: int) -> IoResult<()> {
    // libuv has this as a no-op, so seems like this should as well?
    Ok(())
}

pub fn readlink(p: &CString) -> IoResult<CString> {
    // FIXME: I have a feeling that this reads intermediate symlinks as well.
    use io::c::compat::kernel32::GetFinalPathNameByHandleW;
    let p = try!(to_utf16(p));
    let handle = unsafe {
        libc::CreateFileW(p.as_ptr(),
                          libc::GENERIC_READ,
                          libc::FILE_SHARE_READ,
                          ptr::mut_null(),
                          libc::OPEN_EXISTING,
                          libc::FILE_ATTRIBUTE_NORMAL,
                          ptr::mut_null())
    };
    if handle as int == libc::INVALID_HANDLE_VALUE as int {
        return Err(super::last_error())
    }
    // Specify (sz - 1) because the documentation states that it's the size
    // without the null pointer
    let ret = fill_utf16_buf_and_decode(|buf, sz| unsafe {
        GetFinalPathNameByHandleW(handle,
                                  buf as *u16,
                                  sz - 1,
                                  libc::VOLUME_NAME_DOS)
    });
    let ret = match ret {
        Some(ref s) if s.as_slice().starts_with(r"\\?\") => {
            Ok(Path::new(s.as_slice().slice_from(4)).to_c_str())
        }
        Some(s) => Ok(Path::new(s).to_c_str()),
        None => Err(super::last_error()),
    };
    assert!(unsafe { libc::CloseHandle(handle) } != 0);
    return ret;
}

pub fn symlink(src: &CString, dst: &CString) -> IoResult<()> {
    use io::c::compat::kernel32::CreateSymbolicLinkW;
    let src = try!(to_utf16(src));
    let dst = try!(to_utf16(dst));
    super::mkerr_winbool(unsafe {
        CreateSymbolicLinkW(dst.as_ptr(), src.as_ptr(), 0) as libc::BOOL
    })
}

pub fn link(src: &CString, dst: &CString) -> IoResult<()> {
    let src = try!(to_utf16(src));
    let dst = try!(to_utf16(dst));
    super::mkerr_winbool(unsafe {
        libc::CreateHardLinkW(dst.as_ptr(), src.as_ptr(), ptr::mut_null())
    })
}

fn mkstat(stat: &libc::stat) -> rtio::FileStat {
    rtio::FileStat {
        size: stat.st_size as u64,
        kind: stat.st_mode as u64,
        perm: stat.st_mode as u64,
        created: stat.st_ctime as u64,
        modified: stat.st_mtime as u64,
        accessed: stat.st_atime as u64,
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

pub fn stat(p: &CString) -> IoResult<rtio::FileStat> {
    let mut stat: libc::stat = unsafe { mem::zeroed() };
    let p = try!(to_utf16(p));
    match unsafe { libc::wstat(p.as_ptr(), &mut stat) } {
        0 => Ok(mkstat(&stat)),
        _ => Err(super::last_error()),
    }
}

pub fn lstat(_p: &CString) -> IoResult<rtio::FileStat> {
    // FIXME: implementation is missing
    Err(super::unimpl())
}

pub fn utime(p: &CString, atime: u64, mtime: u64) -> IoResult<()> {
    let buf = libc::utimbuf {
        actime: (atime / 1000) as libc::time64_t,
        modtime: (mtime / 1000) as libc::time64_t,
    };
    let p = try!(to_utf16(p));
    super::mkerr_libc(unsafe {
        libc::wutime(p.as_ptr(), &buf)
    })
}
