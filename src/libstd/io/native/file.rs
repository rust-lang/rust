// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Blocking posix-based file I/O

#[allow(non_camel_case_types)];

use c_str::CString;
use io::IoError;
use io;
use libc::c_int;
use libc;
use ops::Drop;
use option::{Some, None, Option};
use os;
use path::{Path, GenericPath};
use ptr::RawPtr;
use result::{Result, Ok, Err};
use rt::rtio;
use super::IoResult;
use unstable::intrinsics;
use vec::ImmutableVector;
use vec;

#[cfg(windows)] use os::win32::{as_utf16_p, fill_utf16_buf_and_decode};
#[cfg(windows)] use ptr;
#[cfg(windows)] use str;

fn keep_going(data: &[u8], f: |*u8, uint| -> i64) -> i64 {
    #[cfg(windows)] static eintr: int = 0; // doesn't matter
    #[cfg(not(windows))] static eintr: int = libc::EINTR as int;

    let (data, origamt) = do data.as_imm_buf |data, amt| { (data, amt) };
    let mut data = data;
    let mut amt = origamt;
    while amt > 0 {
        let mut ret;
        loop {
            ret = f(data, amt);
            if cfg!(not(windows)) { break } // windows has no eintr
            // if we get an eintr, then try again
            if ret != -1 || os::errno() as int != eintr { break }
        }
        if ret == 0 {
            break
        } else if ret != -1 {
            amt -= ret as uint;
            data = unsafe { data.offset(ret as int) };
        } else {
            return ret;
        }
    }
    return (origamt - amt) as i64;
}

pub type fd_t = libc::c_int;

pub struct FileDesc {
    priv fd: fd_t,
    priv close_on_drop: bool,
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
        FileDesc { fd: fd, close_on_drop: close_on_drop }
    }

    fn inner_read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        #[cfg(windows)] type rlen = libc::c_uint;
        #[cfg(not(windows))] type rlen = libc::size_t;
        let ret = do keep_going(buf) |buf, len| {
            unsafe {
                libc::read(self.fd, buf as *mut libc::c_void, len as rlen) as i64
            }
        };
        if ret == 0 {
            Err(io::standard_error(io::EndOfFile))
        } else if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(ret as uint)
        }
    }
    fn inner_write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        #[cfg(windows)] type wlen = libc::c_uint;
        #[cfg(not(windows))] type wlen = libc::size_t;
        let ret = do keep_going(buf) |buf, len| {
            unsafe {
                libc::write(self.fd, buf as *libc::c_void, len as wlen) as i64
            }
        };
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }
}

impl io::Reader for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match self.inner_read(buf) { Ok(n) => Some(n), Err(*) => None }
    }
    fn eof(&mut self) -> bool { false }
}

impl io::Writer for FileDesc {
    fn write(&mut self, buf: &[u8]) {
        self.inner_write(buf);
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
        return os_pread(self.fd, vec::raw::to_ptr(buf), buf.len(), offset);

        #[cfg(windows)]
        fn os_pread(fd: c_int, buf: *u8, amt: uint, offset: u64) -> IoResult<int> {
            unsafe {
                let mut overlap: libc::OVERLAPPED = intrinsics::init();
                let handle = libc::get_osfhandle(fd) as libc::HANDLE;
                let mut bytes_read = 0;
                overlap.Offset = offset as libc::DWORD;
                overlap.OffsetHigh = (offset >> 32) as libc::DWORD;

                match libc::ReadFile(handle, buf as libc::LPVOID,
                                     amt as libc::DWORD,
                                     &mut bytes_read, &mut overlap) {
                    0 => Err(super::last_error()),
                    _ => Ok(bytes_read as int)
                }
            }
        }

        #[cfg(unix)]
        fn os_pread(fd: c_int, buf: *u8, amt: uint, offset: u64) -> IoResult<int> {
            match unsafe {
                libc::pread(fd, buf as *libc::c_void, amt as libc::size_t,
                            offset as libc::off_t)
            } {
                -1 => Err(super::last_error()),
                n => Ok(n as int)
            }
        }
    }
    fn pwrite(&mut self, buf: &[u8], offset: u64) -> Result<(), IoError> {
        return os_pwrite(self.fd, vec::raw::to_ptr(buf), buf.len(), offset);

        #[cfg(windows)]
        fn os_pwrite(fd: c_int, buf: *u8, amt: uint, offset: u64) -> IoResult<()> {
            unsafe {
                let mut overlap: libc::OVERLAPPED = intrinsics::init();
                let handle = libc::get_osfhandle(fd) as libc::HANDLE;
                overlap.Offset = offset as libc::DWORD;
                overlap.OffsetHigh = (offset >> 32) as libc::DWORD;

                match libc::WriteFile(handle, buf as libc::LPVOID,
                                      amt as libc::DWORD,
                                      ptr::mut_null(), &mut overlap) {
                    0 => Err(super::last_error()),
                    _ => Ok(()),
                }
            }
        }

        #[cfg(unix)]
        fn os_pwrite(fd: c_int, buf: *u8, amt: uint, offset: u64) -> IoResult<()> {
            super::mkerr_libc(unsafe {
                libc::pwrite(fd, buf as *libc::c_void, amt as libc::size_t,
                             offset as libc::off_t)
            } as c_int)
        }
    }
    #[cfg(windows)]
    fn seek(&mut self, pos: i64, style: io::SeekStyle) -> Result<u64, IoError> {
        let whence = match style {
            io::SeekSet => libc::FILE_BEGIN,
            io::SeekEnd => libc::FILE_END,
            io::SeekCur => libc::FILE_CURRENT,
        };
        unsafe {
            let handle = libc::get_osfhandle(self.fd) as libc::HANDLE;
            let mut newpos = 0;
            match libc::SetFilePointerEx(handle, pos, &mut newpos, whence) {
                0 => Err(super::last_error()),
                _ => Ok(newpos as u64),
            }
        }
    }
    #[cfg(unix)]
    fn seek(&mut self, pos: i64, whence: io::SeekStyle) -> Result<u64, IoError> {
        let whence = match whence {
            io::SeekSet => libc::SEEK_SET,
            io::SeekEnd => libc::SEEK_END,
            io::SeekCur => libc::SEEK_CUR,
        };
        let n = unsafe { libc::lseek(self.fd, pos as libc::off_t, whence) };
        if n < 0 {
            Err(super::last_error())
        } else {
            Ok(n as u64)
        }
    }
    fn tell(&self) -> Result<u64, IoError> {
        let n = unsafe { libc::lseek(self.fd, 0, libc::SEEK_CUR) };
        if n < 0 {
            Err(super::last_error())
        } else {
            Ok(n as u64)
        }
    }
    fn fsync(&mut self) -> Result<(), IoError> {
        return os_fsync(self.fd);

        #[cfg(windows)]
        fn os_fsync(fd: c_int) -> IoResult<()> {
            super::mkerr_winbool(unsafe {
                let handle = libc::get_osfhandle(fd);
                libc::FlushFileBuffers(handle as libc::HANDLE)
            })
        }
        #[cfg(unix)]
        fn os_fsync(fd: c_int) -> IoResult<()> {
            super::mkerr_libc(unsafe { libc::fsync(fd) })
        }
    }
    #[cfg(windows)]
    fn datasync(&mut self) -> Result<(), IoError> { return self.fsync(); }

    #[cfg(not(windows))]
    fn datasync(&mut self) -> Result<(), IoError> {
        return super::mkerr_libc(os_datasync(self.fd));

        #[cfg(target_os = "macos")]
        fn os_datasync(fd: c_int) -> c_int {
            unsafe { libc::fcntl(fd, libc::F_FULLFSYNC) }
        }
        #[cfg(target_os = "linux")]
        fn os_datasync(fd: c_int) -> c_int { unsafe { libc::fdatasync(fd) } }
        #[cfg(not(target_os = "macos"), not(target_os = "linux"))]
        fn os_datasync(fd: c_int) -> c_int { unsafe { libc::fsync(fd) } }
    }

    #[cfg(windows)]
    fn truncate(&mut self, offset: i64) -> Result<(), IoError> {
        let orig_pos = match self.tell() { Ok(i) => i, Err(e) => return Err(e) };
        match self.seek(offset, io::SeekSet) {
            Ok(_) => {}, Err(e) => return Err(e),
        };
        let ret = unsafe {
            let handle = libc::get_osfhandle(self.fd) as libc::HANDLE;
            match libc::SetEndOfFile(handle) {
                0 => Err(super::last_error()),
                _ => Ok(())
            }
        };
        self.seek(orig_pos as i64, io::SeekSet);
        return ret;
    }
    #[cfg(unix)]
    fn truncate(&mut self, offset: i64) -> Result<(), IoError> {
        super::mkerr_libc(unsafe {
            libc::ftruncate(self.fd, offset as libc::off_t)
        })
    }
}

impl rtio::RtioPipe for FileDesc {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        self.inner_read(buf)
    }
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        self.inner_write(buf)
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

impl Drop for FileDesc {
    fn drop(&mut self) {
        // closing stdio file handles makes no sense, so never do it
        if self.close_on_drop && self.fd > libc::STDERR_FILENO {
            unsafe { libc::close(self.fd); }
        }
    }
}

pub struct CFile {
    priv file: *libc::FILE,
    priv fd: FileDesc,
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
        super::mkerr_libc(unsafe { libc::fflush(self.file) })
    }
}

impl rtio::RtioFileStream for CFile {
    fn read(&mut self, buf: &mut [u8]) -> Result<int, IoError> {
        let ret = do keep_going(buf) |buf, len| {
            unsafe {
                libc::fread(buf as *mut libc::c_void, 1, len as libc::size_t,
                            self.file) as i64
            }
        };
        if ret == 0 {
            Err(io::standard_error(io::EndOfFile))
        } else if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(ret as int)
        }
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let ret = do keep_going(buf) |buf, len| {
            unsafe {
                libc::fwrite(buf as *libc::c_void, 1, len as libc::size_t,
                            self.file) as i64
            }
        };
        if ret < 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }

    fn pread(&mut self, buf: &mut [u8], offset: u64) -> Result<int, IoError> {
        self.flush();
        self.fd.pread(buf, offset)
    }
    fn pwrite(&mut self, buf: &[u8], offset: u64) -> Result<(), IoError> {
        self.flush();
        self.fd.pwrite(buf, offset)
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
        self.flush();
        self.fd.fsync()
    }
    fn datasync(&mut self) -> Result<(), IoError> {
        self.flush();
        self.fd.fsync()
    }
    fn truncate(&mut self, offset: i64) -> Result<(), IoError> {
        self.flush();
        self.fd.truncate(offset)
    }
}

impl Drop for CFile {
    fn drop(&mut self) {
        unsafe { libc::fclose(self.file); }
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

    return match os_open(path, flags, mode) {
        -1 => Err(super::last_error()),
        fd => Ok(FileDesc::new(fd, true)),
    };

    #[cfg(windows)]
    fn os_open(path: &CString, flags: c_int, mode: c_int) -> c_int {
        do as_utf16_p(path.as_str().unwrap()) |path| {
            unsafe { libc::wopen(path, flags, mode) }
        }
    }

    #[cfg(unix)]
    fn os_open(path: &CString, flags: c_int, mode: c_int) -> c_int {
        unsafe { libc::open(path.with_ref(|p| p), flags, mode) }
    }
}

pub fn mkdir(p: &CString, mode: io::FilePermission) -> IoResult<()> {
    return os_mkdir(p, mode as c_int);

    #[cfg(windows)]
    fn os_mkdir(p: &CString, _mode: c_int) -> IoResult<()> {
        super::mkerr_winbool(unsafe {
            // FIXME: turn mode into something useful? #2623
            do as_utf16_p(p.as_str().unwrap()) |buf| {
                libc::CreateDirectoryW(buf, ptr::mut_null())
            }
        })
    }

    #[cfg(unix)]
    fn os_mkdir(p: &CString, mode: c_int) -> IoResult<()> {
        super::mkerr_libc(unsafe {
            libc::mkdir(p.with_ref(|p| p), mode as libc::mode_t)
        })
    }
}

pub fn readdir(p: &CString) -> IoResult<~[Path]> {
    fn prune(root: &CString, dirs: ~[Path]) -> ~[Path] {
        let root = unsafe { CString::new(root.with_ref(|p| p), false) };
        let root = Path::new(root);

        dirs.move_iter().filter(|path| {
            path.as_vec() != bytes!(".") && path.as_vec() != bytes!("..")
        }).map(|path| root.join(path)).collect()
    }

    unsafe {
        #[cfg(not(windows))]
        unsafe fn get_list(p: &CString) -> IoResult<~[Path]> {
            use libc::{dirent_t};
            use libc::{opendir, readdir, closedir};
            extern {
                fn rust_list_dir_val(ptr: *dirent_t) -> *libc::c_char;
            }
            debug!("os::list_dir -- BEFORE OPENDIR");

            let dir_ptr = do p.with_ref |buf| {
                opendir(buf)
            };

            if (dir_ptr as uint != 0) {
                let mut paths = ~[];
                debug!("os::list_dir -- opendir() SUCCESS");
                let mut entry_ptr = readdir(dir_ptr);
                while (entry_ptr as uint != 0) {
                    let cstr = CString::new(rust_list_dir_val(entry_ptr), false);
                    paths.push(Path::new(cstr));
                    entry_ptr = readdir(dir_ptr);
                }
                closedir(dir_ptr);
                Ok(paths)
            } else {
                Err(super::last_error())
            }
        }

        #[cfg(windows)]
        unsafe fn get_list(p: &CString) -> IoResult<~[Path]> {
            use libc::consts::os::extra::INVALID_HANDLE_VALUE;
            use libc::{wcslen, free};
            use libc::funcs::extra::kernel32::{
                FindFirstFileW,
                FindNextFileW,
                FindClose,
            };
            use libc::types::os::arch::extra::HANDLE;
            use os::win32::{
                as_utf16_p
            };
            use rt::global_heap::malloc_raw;

            #[nolink]
            extern {
                fn rust_list_dir_wfd_size() -> libc::size_t;
                fn rust_list_dir_wfd_fp_buf(wfd: *libc::c_void) -> *u16;
            }
            let p = CString::new(p.with_ref(|p| p), false);
            let p = Path::new(p);
            let star = p.join("*");
            do as_utf16_p(star.as_str().unwrap()) |path_ptr| {
                let wfd_ptr = malloc_raw(rust_list_dir_wfd_size() as uint);
                let find_handle = FindFirstFileW(path_ptr, wfd_ptr as HANDLE);
                if find_handle as libc::c_int != INVALID_HANDLE_VALUE {
                    let mut paths = ~[];
                    let mut more_files = 1 as libc::c_int;
                    while more_files != 0 {
                        let fp_buf = rust_list_dir_wfd_fp_buf(wfd_ptr);
                        if fp_buf as uint == 0 {
                            fail!("os::list_dir() failure: got null ptr from wfd");
                        }
                        else {
                            let fp_vec = vec::from_buf(
                                fp_buf, wcslen(fp_buf) as uint);
                            let fp_str = str::from_utf16(fp_vec);
                            paths.push(Path::new(fp_str));
                        }
                        more_files = FindNextFileW(find_handle, wfd_ptr as HANDLE);
                    }
                    FindClose(find_handle);
                    free(wfd_ptr);
                    Ok(paths)
                } else {
                    Err(super::last_error())
                }
            }
        }

        get_list(p).map(|paths| prune(p, paths))
    }
}

pub fn unlink(p: &CString) -> IoResult<()> {
    return os_unlink(p);

    #[cfg(windows)]
    fn os_unlink(p: &CString) -> IoResult<()> {
        super::mkerr_winbool(unsafe {
            do as_utf16_p(p.as_str().unwrap()) |buf| {
                libc::DeleteFileW(buf)
            }
        })
    }

    #[cfg(unix)]
    fn os_unlink(p: &CString) -> IoResult<()> {
        super::mkerr_libc(unsafe { libc::unlink(p.with_ref(|p| p)) })
    }
}

pub fn rename(old: &CString, new: &CString) -> IoResult<()> {
    return os_rename(old, new);

    #[cfg(windows)]
    fn os_rename(old: &CString, new: &CString) -> IoResult<()> {
        super::mkerr_winbool(unsafe {
            do as_utf16_p(old.as_str().unwrap()) |old| {
                do as_utf16_p(new.as_str().unwrap()) |new| {
                    libc::MoveFileExW(old, new, libc::MOVEFILE_REPLACE_EXISTING)
                }
            }
        })
    }

    #[cfg(unix)]
    fn os_rename(old: &CString, new: &CString) -> IoResult<()> {
        super::mkerr_libc(unsafe {
            libc::rename(old.with_ref(|p| p), new.with_ref(|p| p))
        })
    }
}

pub fn chmod(p: &CString, mode: io::FilePermission) -> IoResult<()> {
    return super::mkerr_libc(os_chmod(p, mode as c_int));

    #[cfg(windows)]
    fn os_chmod(p: &CString, mode: c_int) -> c_int {
        unsafe {
            do as_utf16_p(p.as_str().unwrap()) |p| {
                libc::wchmod(p, mode)
            }
        }
    }

    #[cfg(unix)]
    fn os_chmod(p: &CString, mode: c_int) -> c_int {
        unsafe { libc::chmod(p.with_ref(|p| p), mode as libc::mode_t) }
    }
}

pub fn rmdir(p: &CString) -> IoResult<()> {
    return super::mkerr_libc(os_rmdir(p));

    #[cfg(windows)]
    fn os_rmdir(p: &CString) -> c_int {
        unsafe {
            do as_utf16_p(p.as_str().unwrap()) |p| { libc::wrmdir(p) }
        }
    }

    #[cfg(unix)]
    fn os_rmdir(p: &CString) -> c_int {
        unsafe { libc::rmdir(p.with_ref(|p| p)) }
    }
}

pub fn chown(p: &CString, uid: int, gid: int) -> IoResult<()> {
    return super::mkerr_libc(os_chown(p, uid, gid));

    // libuv has this as a no-op, so seems like this should as well?
    #[cfg(windows)]
    fn os_chown(_p: &CString, _uid: int, _gid: int) -> c_int { 0 }

    #[cfg(unix)]
    fn os_chown(p: &CString, uid: int, gid: int) -> c_int {
        unsafe {
            libc::chown(p.with_ref(|p| p), uid as libc::uid_t,
                        gid as libc::gid_t)
        }
    }
}

pub fn readlink(p: &CString) -> IoResult<Path> {
    return os_readlink(p);

    // XXX: I have a feeling that this reads intermediate symlinks as well.
    #[cfg(windows)]
    fn os_readlink(p: &CString) -> IoResult<Path> {
        let handle = unsafe {
            do as_utf16_p(p.as_str().unwrap()) |p| {
                libc::CreateFileW(p,
                                  libc::GENERIC_READ,
                                  libc::FILE_SHARE_READ,
                                  ptr::mut_null(),
                                  libc::OPEN_EXISTING,
                                  libc::FILE_ATTRIBUTE_NORMAL,
                                  ptr::mut_null())
            }
        };
        if handle == ptr::mut_null() { return Err(super::last_error()) }
        let ret = do fill_utf16_buf_and_decode |buf, sz| {
            unsafe {
                libc::GetFinalPathNameByHandleW(handle, buf as *u16, sz,
                                                libc::VOLUME_NAME_NT)
            }
        };
        let ret = match ret {
            Some(s) => Ok(Path::new(s)),
            None => Err(super::last_error()),
        };
        unsafe { libc::CloseHandle(handle) };
        return ret;

    }

    #[cfg(unix)]
    fn os_readlink(p: &CString) -> IoResult<Path> {
        let p = p.with_ref(|p| p);
        let mut len = unsafe { libc::pathconf(p, libc::_PC_NAME_MAX) };
        if len == -1 {
            len = 1024; // XXX: read PATH_MAX from C ffi?
        }
        let mut buf = vec::with_capacity::<u8>(len as uint);
        match unsafe {
            libc::readlink(p, vec::raw::to_ptr(buf) as *mut libc::c_char,
                           len as libc::size_t)
        } {
            -1 => Err(super::last_error()),
            n => {
                assert!(n > 0);
                unsafe { vec::raw::set_len(&mut buf, n as uint); }
                Ok(Path::new(buf))
            }
        }
    }
}

pub fn symlink(src: &CString, dst: &CString) -> IoResult<()> {
    return os_symlink(src, dst);

    #[cfg(windows)]
    fn os_symlink(src: &CString, dst: &CString) -> IoResult<()> {
        super::mkerr_winbool(do as_utf16_p(src.as_str().unwrap()) |src| {
            do as_utf16_p(dst.as_str().unwrap()) |dst| {
                unsafe { libc::CreateSymbolicLinkW(dst, src, 0) }
            }
        })
    }

    #[cfg(unix)]
    fn os_symlink(src: &CString, dst: &CString) -> IoResult<()> {
        super::mkerr_libc(unsafe {
            libc::symlink(src.with_ref(|p| p), dst.with_ref(|p| p))
        })
    }
}

pub fn link(src: &CString, dst: &CString) -> IoResult<()> {
    return os_link(src, dst);

    #[cfg(windows)]
    fn os_link(src: &CString, dst: &CString) -> IoResult<()> {
        super::mkerr_winbool(do as_utf16_p(src.as_str().unwrap()) |src| {
            do as_utf16_p(dst.as_str().unwrap()) |dst| {
                unsafe { libc::CreateHardLinkW(dst, src, ptr::mut_null()) }
            }
        })
    }

    #[cfg(unix)]
    fn os_link(src: &CString, dst: &CString) -> IoResult<()> {
        super::mkerr_libc(unsafe {
            libc::link(src.with_ref(|p| p), dst.with_ref(|p| p))
        })
    }
}

#[cfg(windows)]
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

    io::FileStat {
        path: Path::new(path),
        size: stat.st_size as u64,
        kind: kind,
        perm: (stat.st_mode) as io::FilePermission & io::AllPermissions,
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

#[cfg(unix)]
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
        perm: (stat.st_mode) as io::FilePermission & io::AllPermissions,
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
    return os_stat(p);

    #[cfg(windows)]
    fn os_stat(p: &CString) -> IoResult<io::FileStat> {
        let mut stat: libc::stat = unsafe { intrinsics::uninit() };
        do as_utf16_p(p.as_str().unwrap()) |up| {
            match unsafe { libc::wstat(up, &mut stat) } {
                0 => Ok(mkstat(&stat, p)),
                _ => Err(super::last_error()),
            }
        }
    }

    #[cfg(unix)]
    fn os_stat(p: &CString) -> IoResult<io::FileStat> {
        let mut stat: libc::stat = unsafe { intrinsics::uninit() };
        match unsafe { libc::stat(p.with_ref(|p| p), &mut stat) } {
            0 => Ok(mkstat(&stat, p)),
            _ => Err(super::last_error()),
        }
    }
}

pub fn lstat(p: &CString) -> IoResult<io::FileStat> {
    return os_lstat(p);

    // XXX: windows implementation is missing
    #[cfg(windows)]
    fn os_lstat(_p: &CString) -> IoResult<io::FileStat> {
        Err(super::unimpl())
    }

    #[cfg(unix)]
    fn os_lstat(p: &CString) -> IoResult<io::FileStat> {
        let mut stat: libc::stat = unsafe { intrinsics::uninit() };
        match unsafe { libc::lstat(p.with_ref(|p| p), &mut stat) } {
            0 => Ok(mkstat(&stat, p)),
            _ => Err(super::last_error()),
        }
    }
}

pub fn utime(p: &CString, atime: u64, mtime: u64) -> IoResult<()> {
    return super::mkerr_libc(os_utime(p, atime, mtime));

    #[cfg(windows)]
    fn os_utime(p: &CString, atime: u64, mtime: u64) -> c_int {
        let buf = libc::utimbuf {
            actime: (atime / 1000) as libc::time64_t,
            modtime: (mtime / 1000) as libc::time64_t,
        };
        unsafe {
            do as_utf16_p(p.as_str().unwrap()) |p| {
                libc::wutime(p, &buf)
            }
        }
    }

    #[cfg(unix)]
    fn os_utime(p: &CString, atime: u64, mtime: u64) -> c_int {
        let buf = libc::utimbuf {
            actime: (atime / 1000) as libc::time_t,
            modtime: (mtime / 1000) as libc::time_t,
        };
        unsafe { libc::utime(p.with_ref(|p| p), &buf) }
    }
}

#[cfg(test)]
mod tests {
    use io::native::file::{CFile, FileDesc};
    use io::fs;
    use io;
    use libc;
    use os;
    use path::Path;
    use rand;
    use result::Ok;
    use rt::rtio::RtioFileStream;

    fn tmpdir() -> Path {
        let ret = os::tmpdir().join(format!("rust-{}", rand::random::<u32>()));
        fs::mkdir(&ret, io::UserRWX);
        ret
    }

    #[ignore(cfg(target_os = "freebsd"))] // hmm, maybe pipes have a tiny buffer
    fn test_file_desc() {
        // Run this test with some pipes so we don't have to mess around with
        // opening or closing files.
        unsafe {
            let os::Pipe { input, out } = os::pipe();
            let mut reader = FileDesc::new(input, true);
            let mut writer = FileDesc::new(out, true);

            writer.inner_write(bytes!("test"));
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
    }

    #[ignore(cfg(windows))] // apparently windows doesn't like tmpfile
    fn test_cfile() {
        unsafe {
            let f = libc::tmpfile();
            assert!(!f.is_null());
            let mut file = CFile::new(f);

            file.write(bytes!("test"));
            let mut buf = [0u8, ..4];
            file.seek(0, io::SeekSet);
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
