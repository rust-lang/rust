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

use io::IoError;
use io;
use libc;
use ops::Drop;
use option::{Some, None, Option};
use os;
use ptr::RawPtr;
use result::{Result, Ok, Err};
use rt::rtio;
use vec::ImmutableVector;

fn keep_going(data: &[u8], f: &fn(*u8, uint) -> i64) -> i64 {
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
    fn pread(&mut self, _buf: &mut [u8], _offset: u64) -> Result<int, IoError> {
        Err(super::unimpl())
    }
    fn pwrite(&mut self, _buf: &[u8], _offset: u64) -> Result<(), IoError> {
        Err(super::unimpl())
    }
    fn seek(&mut self, _pos: i64, _whence: io::SeekStyle) -> Result<u64, IoError> {
        Err(super::unimpl())
    }
    fn tell(&self) -> Result<u64, IoError> {
        Err(super::unimpl())
    }
    fn fsync(&mut self) -> Result<(), IoError> {
        Err(super::unimpl())
    }
    fn datasync(&mut self) -> Result<(), IoError> {
        Err(super::unimpl())
    }
    fn truncate(&mut self, _offset: i64) -> Result<(), IoError> {
        Err(super::unimpl())
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
    priv file: *libc::FILE
}

impl CFile {
    /// Create a `CFile` from an open `FILE` pointer.
    ///
    /// The `CFile` takes ownership of the `FILE` pointer and will close it upon
    /// destruction.
    pub fn new(file: *libc::FILE) -> CFile { CFile { file: file } }
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

    fn pread(&mut self, _buf: &mut [u8], _offset: u64) -> Result<int, IoError> {
        Err(super::unimpl())
    }
    fn pwrite(&mut self, _buf: &[u8], _offset: u64) -> Result<(), IoError> {
        Err(super::unimpl())
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
        Err(super::unimpl())
    }
    fn datasync(&mut self) -> Result<(), IoError> {
        Err(super::unimpl())
    }
    fn truncate(&mut self, _offset: i64) -> Result<(), IoError> {
        Err(super::unimpl())
    }
}

impl Drop for CFile {
    fn drop(&mut self) {
        unsafe { libc::fclose(self.file); }
    }
}

#[cfg(test)]
mod tests {
    use libc;
    use os;
    use io::{io_error, SeekSet, Writer, Reader};
    use result::Ok;
    use super::{CFile, FileDesc};

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
            file.seek(0, SeekSet);
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

// n.b. these functions were all part of the old `std::os` module. There's lots
//      of fun little nuances that were taken care of by these functions, but
//      they are all thread-blocking versions that are no longer desired (we now
//      use a non-blocking event loop implementation backed by libuv).
//
//      In theory we will have a thread-blocking version of the event loop (if
//      desired), so these functions may just need to get adapted to work in
//      those situtations. For now, I'm leaving the code around so it doesn't
//      get bitrotted instantaneously.
mod old_os {
    use prelude::*;
    use libc::{size_t, c_void, c_int};
    use libc;
    use vec;

    #[cfg(not(windows))] use c_str::CString;
    #[cfg(not(windows))] use libc::fclose;
    #[cfg(test)] #[cfg(windows)] use os;
    #[cfg(test)] use rand;
    #[cfg(windows)] use str;
    #[cfg(windows)] use ptr;

    // On Windows, wide character version of function must be used to support
    // unicode, so functions should be split into at least two versions,
    // which are for Windows and for non-Windows, if necessary.
    // See https://github.com/mozilla/rust/issues/9822 for more information.

    mod rustrt {
        use libc::{c_char, c_int};
        use libc;

        extern {
            pub fn rust_path_is_dir(path: *libc::c_char) -> c_int;
            pub fn rust_path_exists(path: *libc::c_char) -> c_int;
        }

        // Uses _wstat instead of stat.
        #[cfg(windows)]
        extern {
            pub fn rust_path_is_dir_u16(path: *u16) -> c_int;
            pub fn rust_path_exists_u16(path: *u16) -> c_int;
        }
    }

    /// Recursively walk a directory structure
    pub fn walk_dir(p: &Path, f: &fn(&Path) -> bool) -> bool {
        let r = list_dir(p);
        r.iter().advance(|q| {
            let path = &p.join(q);
            f(path) && (!path_is_dir(path) || walk_dir(path, |p| f(p)))
        })
    }

    #[cfg(unix)]
    /// Indicates whether a path represents a directory
    pub fn path_is_dir(p: &Path) -> bool {
        unsafe {
            do p.with_c_str |buf| {
                rustrt::rust_path_is_dir(buf) != 0 as c_int
            }
        }
    }


    #[cfg(windows)]
    pub fn path_is_dir(p: &Path) -> bool {
        unsafe {
            do os::win32::as_utf16_p(p.as_str().unwrap()) |buf| {
                rustrt::rust_path_is_dir_u16(buf) != 0 as c_int
            }
        }
    }

    #[cfg(unix)]
    /// Indicates whether a path exists
    pub fn path_exists(p: &Path) -> bool {
        unsafe {
            do p.with_c_str |buf| {
                rustrt::rust_path_exists(buf) != 0 as c_int
            }
        }
    }

    #[cfg(windows)]
    pub fn path_exists(p: &Path) -> bool {
        unsafe {
            do os::win32::as_utf16_p(p.as_str().unwrap()) |buf| {
                rustrt::rust_path_exists_u16(buf) != 0 as c_int
            }
        }
    }

    /// Creates a directory at the specified path
    pub fn make_dir(p: &Path, mode: c_int) -> bool {
        return mkdir(p, mode);

        #[cfg(windows)]
        fn mkdir(p: &Path, _mode: c_int) -> bool {
            unsafe {
                use os::win32::as_utf16_p;
                // FIXME: turn mode into something useful? #2623
                do as_utf16_p(p.as_str().unwrap()) |buf| {
                    libc::CreateDirectoryW(buf, ptr::mut_null())
                        != (0 as libc::BOOL)
                }
            }
        }

        #[cfg(unix)]
        fn mkdir(p: &Path, mode: c_int) -> bool {
            do p.with_c_str |buf| {
                unsafe {
                    libc::mkdir(buf, mode as libc::mode_t) == (0 as c_int)
                }
            }
        }
    }

    /// Creates a directory with a given mode.
    /// Returns true iff creation
    /// succeeded. Also creates all intermediate subdirectories
    /// if they don't already exist, giving all of them the same mode.

    // tjc: if directory exists but with different permissions,
    // should we return false?
    pub fn mkdir_recursive(p: &Path, mode: c_int) -> bool {
        if path_is_dir(p) {
            return true;
        }
        if p.filename().is_some() {
            let mut p_ = p.clone();
            p_.pop();
            if !mkdir_recursive(&p_, mode) {
                return false;
            }
        }
        return make_dir(p, mode);
    }

    /// Lists the contents of a directory
    ///
    /// Each resulting Path is a relative path with no directory component.
    pub fn list_dir(p: &Path) -> ~[Path] {
        unsafe {
            #[cfg(target_os = "linux")]
            #[cfg(target_os = "android")]
            #[cfg(target_os = "freebsd")]
            #[cfg(target_os = "macos")]
            unsafe fn get_list(p: &Path) -> ~[Path] {
                use libc::{dirent_t};
                use libc::{opendir, readdir, closedir};
                extern {
                    fn rust_list_dir_val(ptr: *dirent_t) -> *libc::c_char;
                }
                let mut paths = ~[];
                debug!("os::list_dir -- BEFORE OPENDIR");

                let dir_ptr = do p.with_c_str |buf| {
                    opendir(buf)
                };

                if (dir_ptr as uint != 0) {
                    debug!("os::list_dir -- opendir() SUCCESS");
                    let mut entry_ptr = readdir(dir_ptr);
                    while (entry_ptr as uint != 0) {
                        let cstr = CString::new(rust_list_dir_val(entry_ptr), false);
                        paths.push(Path::new(cstr));
                        entry_ptr = readdir(dir_ptr);
                    }
                    closedir(dir_ptr);
                }
                else {
                    debug!("os::list_dir -- opendir() FAILURE");
                }
                debug!("os::list_dir -- AFTER -- \\#: {}", paths.len());
                paths
            }
            #[cfg(windows)]
            unsafe fn get_list(p: &Path) -> ~[Path] {
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
                let star = p.join("*");
                do as_utf16_p(star.as_str().unwrap()) |path_ptr| {
                    let mut paths = ~[];
                    let wfd_ptr = malloc_raw(rust_list_dir_wfd_size() as uint);
                    let find_handle = FindFirstFileW(path_ptr, wfd_ptr as HANDLE);
                    if find_handle as libc::c_int != INVALID_HANDLE_VALUE {
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
                        free(wfd_ptr)
                    }
                    paths
                }
            }
            do get_list(p).move_iter().filter |path| {
                path.as_vec() != bytes!(".") && path.as_vec() != bytes!("..")
            }.collect()
        }
    }

    /// Removes a directory at the specified path, after removing
    /// all its contents. Use carefully!
    pub fn remove_dir_recursive(p: &Path) -> bool {
        let mut error_happened = false;
        do walk_dir(p) |inner| {
            if !error_happened {
                if path_is_dir(inner) {
                    if !remove_dir_recursive(inner) {
                        error_happened = true;
                    }
                }
                else {
                    if !remove_file(inner) {
                        error_happened = true;
                    }
                }
            }
            true
        };
        // Directory should now be empty
        !error_happened && remove_dir(p)
    }

    /// Removes a directory at the specified path
    pub fn remove_dir(p: &Path) -> bool {
       return rmdir(p);

        #[cfg(windows)]
        fn rmdir(p: &Path) -> bool {
            unsafe {
                use os::win32::as_utf16_p;
                return do as_utf16_p(p.as_str().unwrap()) |buf| {
                    libc::RemoveDirectoryW(buf) != (0 as libc::BOOL)
                };
            }
        }

        #[cfg(unix)]
        fn rmdir(p: &Path) -> bool {
            do p.with_c_str |buf| {
                unsafe {
                    libc::rmdir(buf) == (0 as c_int)
                }
            }
        }
    }

    /// Deletes an existing file
    pub fn remove_file(p: &Path) -> bool {
        return unlink(p);

        #[cfg(windows)]
        fn unlink(p: &Path) -> bool {
            unsafe {
                use os::win32::as_utf16_p;
                return do as_utf16_p(p.as_str().unwrap()) |buf| {
                    libc::DeleteFileW(buf) != (0 as libc::BOOL)
                };
            }
        }

        #[cfg(unix)]
        fn unlink(p: &Path) -> bool {
            unsafe {
                do p.with_c_str |buf| {
                    libc::unlink(buf) == (0 as c_int)
                }
            }
        }
    }

    /// Renames an existing file or directory
    pub fn rename_file(old: &Path, new: &Path) -> bool {
        unsafe {
           do old.with_c_str |old_buf| {
                do new.with_c_str |new_buf| {
                    libc::rename(old_buf, new_buf) == (0 as c_int)
                }
           }
        }
    }

    /// Copies a file from one location to another
    pub fn copy_file(from: &Path, to: &Path) -> bool {
        return do_copy_file(from, to);

        #[cfg(windows)]
        fn do_copy_file(from: &Path, to: &Path) -> bool {
            unsafe {
                use os::win32::as_utf16_p;
                return do as_utf16_p(from.as_str().unwrap()) |fromp| {
                    do as_utf16_p(to.as_str().unwrap()) |top| {
                        libc::CopyFileW(fromp, top, (0 as libc::BOOL)) !=
                            (0 as libc::BOOL)
                    }
                }
            }
        }

        #[cfg(unix)]
        fn do_copy_file(from: &Path, to: &Path) -> bool {
            unsafe {
                let istream = do from.with_c_str |fromp| {
                    do "rb".with_c_str |modebuf| {
                        libc::fopen(fromp, modebuf)
                    }
                };
                if istream as uint == 0u {
                    return false;
                }
                // Preserve permissions
                let from_mode = from.stat().perm;

                let ostream = do to.with_c_str |top| {
                    do "w+b".with_c_str |modebuf| {
                        libc::fopen(top, modebuf)
                    }
                };
                if ostream as uint == 0u {
                    fclose(istream);
                    return false;
                }
                let bufsize = 8192u;
                let mut buf = vec::with_capacity::<u8>(bufsize);
                let mut done = false;
                let mut ok = true;
                while !done {
                    do buf.as_mut_buf |b, _sz| {
                      let nread = libc::fread(b as *mut c_void, 1u as size_t,
                                              bufsize as size_t,
                                              istream);
                      if nread > 0 as size_t {
                          if libc::fwrite(b as *c_void, 1u as size_t, nread,
                                          ostream) != nread {
                              ok = false;
                              done = true;
                          }
                      } else {
                          done = true;
                      }
                  }
                }
                fclose(istream);
                fclose(ostream);

                // Give the new file the old file's permissions
                if do to.with_c_str |to_buf| {
                    libc::chmod(to_buf, from_mode as libc::mode_t)
                } != 0 {
                    return false; // should be a condition...
                }
                return ok;
            }
        }
    }

    #[test]
    fn tmpdir() {
        let p = os::tmpdir();
        let s = p.as_str();
        assert!(s.is_some() && s.unwrap() != ".");
    }

    // Issue #712
    #[test]
    fn test_list_dir_no_invalid_memory_access() {
        list_dir(&Path::new("."));
    }

    #[test]
    fn test_list_dir() {
        let dirs = list_dir(&Path::new("."));
        // Just assuming that we've got some contents in the current directory
        assert!(dirs.len() > 0u);

        for dir in dirs.iter() {
            debug!("{:?}", (*dir).clone());
        }
    }

    #[test]
    #[cfg(not(windows))]
    fn test_list_dir_root() {
        let dirs = list_dir(&Path::new("/"));
        assert!(dirs.len() > 1);
    }
    #[test]
    #[cfg(windows)]
    fn test_list_dir_root() {
        let dirs = list_dir(&Path::new("C:\\"));
        assert!(dirs.len() > 1);
    }

    #[test]
    fn test_path_is_dir() {
        use io::fs::{mkdir_recursive};
        use io::{File, UserRWX};

        assert!((path_is_dir(&Path::new("."))));
        assert!((!path_is_dir(&Path::new("test/stdtest/fs.rs"))));

        let mut dirpath = os::tmpdir();
        dirpath.push(format!("rust-test-{}/test-\uac00\u4e00\u30fc\u4f60\u597d",
            rand::random::<u32>())); // 가一ー你好
        debug!("path_is_dir dirpath: {}", dirpath.display());

        mkdir_recursive(&dirpath, UserRWX);

        assert!((path_is_dir(&dirpath)));

        let mut filepath = dirpath;
        filepath.push("unicode-file-\uac00\u4e00\u30fc\u4f60\u597d.rs");
        debug!("path_is_dir filepath: {}", filepath.display());

        File::create(&filepath); // ignore return; touch only
        assert!((!path_is_dir(&filepath)));

        assert!((!path_is_dir(&Path::new(
                     "test/unicode-bogus-dir-\uac00\u4e00\u30fc\u4f60\u597d"))));
    }

    #[test]
    fn test_path_exists() {
        use io::fs::mkdir_recursive;
        use io::UserRWX;

        assert!((path_exists(&Path::new("."))));
        assert!((!path_exists(&Path::new(
                     "test/nonexistent-bogus-path"))));

        let mut dirpath = os::tmpdir();
        dirpath.push(format!("rust-test-{}/test-\uac01\u4e01\u30fc\u518d\u89c1",
            rand::random::<u32>())); // 각丁ー再见

        mkdir_recursive(&dirpath, UserRWX);
        assert!((path_exists(&dirpath)));
        assert!((!path_exists(&Path::new(
                     "test/unicode-bogus-path-\uac01\u4e01\u30fc\u518d\u89c1"))));
    }
}
