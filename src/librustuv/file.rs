// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::c_str::CString;
use std::c_str;
use std::cast::transmute;
use std::cast;
use std::libc::{c_int, c_char, c_void, size_t};
use std::libc;
use std::rt::task::BlockedTask;
use std::io::{FileStat, IoError};
use std::io;
use std::rt::rtio;

use homing::{HomingIO, HomeHandle};
use super::{Loop, UvError, uv_error_to_io_error, wait_until_woken_after, wakeup};
use uvio::UvIoFactory;
use uvll;

pub struct FsRequest {
    req: *uvll::uv_fs_t,
    priv fired: bool,
}

pub struct FileWatcher {
    priv loop_: Loop,
    priv fd: c_int,
    priv close: rtio::CloseBehavior,
    priv home: HomeHandle,
}

impl FsRequest {
    pub fn open(io: &mut UvIoFactory, path: &CString, flags: int, mode: int)
        -> Result<FileWatcher, UvError>
    {
        execute(|req, cb| unsafe {
            uvll::uv_fs_open(io.uv_loop(),
                             req, path.with_ref(|p| p), flags as c_int,
                             mode as c_int, cb)
        }).map(|req|
            FileWatcher::new(io, req.get_result() as c_int,
                             rtio::CloseSynchronously)
        )
    }

    pub fn unlink(loop_: &Loop, path: &CString) -> Result<(), UvError> {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_unlink(loop_.handle, req, path.with_ref(|p| p),
                               cb)
        })
    }

    pub fn lstat(loop_: &Loop, path: &CString) -> Result<FileStat, UvError> {
        execute(|req, cb| unsafe {
            uvll::uv_fs_lstat(loop_.handle, req, path.with_ref(|p| p),
                              cb)
        }).map(|req| req.mkstat())
    }

    pub fn stat(loop_: &Loop, path: &CString) -> Result<FileStat, UvError> {
        execute(|req, cb| unsafe {
            uvll::uv_fs_stat(loop_.handle, req, path.with_ref(|p| p),
                             cb)
        }).map(|req| req.mkstat())
    }

    pub fn write(loop_: &Loop, fd: c_int, buf: &[u8], offset: i64)
        -> Result<(), UvError>
    {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_write(loop_.handle, req,
                              fd, buf.as_ptr() as *c_void,
                              buf.len() as size_t, offset, cb)
        })
    }

    pub fn read(loop_: &Loop, fd: c_int, buf: &mut [u8], offset: i64)
        -> Result<int, UvError>
    {
        execute(|req, cb| unsafe {
            uvll::uv_fs_read(loop_.handle, req,
                             fd, buf.as_ptr() as *c_void,
                             buf.len() as size_t, offset, cb)
        }).map(|req| {
            req.get_result() as int
        })
    }

    pub fn mkdir(loop_: &Loop, path: &CString, mode: c_int)
        -> Result<(), UvError>
    {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_mkdir(loop_.handle, req, path.with_ref(|p| p),
                              mode, cb)
        })
    }

    pub fn rmdir(loop_: &Loop, path: &CString) -> Result<(), UvError> {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_rmdir(loop_.handle, req, path.with_ref(|p| p),
                              cb)
        })
    }

    pub fn rename(loop_: &Loop, path: &CString, to: &CString)
        -> Result<(), UvError>
    {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_rename(loop_.handle,
                               req,
                               path.with_ref(|p| p),
                               to.with_ref(|p| p),
                               cb)
        })
    }

    pub fn chmod(loop_: &Loop, path: &CString, mode: c_int)
        -> Result<(), UvError>
    {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_chmod(loop_.handle, req, path.with_ref(|p| p),
                              mode, cb)
        })
    }

    pub fn readdir(loop_: &Loop, path: &CString, flags: c_int)
        -> Result<~[Path], UvError>
    {
        execute(|req, cb| unsafe {
            uvll::uv_fs_readdir(loop_.handle,
                                req, path.with_ref(|p| p), flags, cb)
        }).map(|req| unsafe {
            let mut paths = ~[];
            let path = CString::new(path.with_ref(|p| p), false);
            let parent = Path::new(path);
            let _ = c_str::from_c_multistring(req.get_ptr() as *libc::c_char,
                                              Some(req.get_result() as uint),
                                              |rel| {
                let p = rel.as_bytes();
                paths.push(parent.join(p.slice_to(rel.len())));
            });
            paths
        })
    }

    pub fn readlink(loop_: &Loop, path: &CString) -> Result<Path, UvError> {
        execute(|req, cb| unsafe {
            uvll::uv_fs_readlink(loop_.handle, req,
                                 path.with_ref(|p| p), cb)
        }).map(|req| {
            Path::new(unsafe {
                CString::new(req.get_ptr() as *libc::c_char, false)
            })
        })
    }

    pub fn chown(loop_: &Loop, path: &CString, uid: int, gid: int)
        -> Result<(), UvError>
    {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_chown(loop_.handle,
                              req, path.with_ref(|p| p),
                              uid as uvll::uv_uid_t,
                              gid as uvll::uv_gid_t,
                              cb)
        })
    }

    pub fn truncate(loop_: &Loop, file: c_int, offset: i64)
        -> Result<(), UvError>
    {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_ftruncate(loop_.handle, req, file, offset, cb)
        })
    }

    pub fn link(loop_: &Loop, src: &CString, dst: &CString)
        -> Result<(), UvError>
    {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_link(loop_.handle, req,
                             src.with_ref(|p| p),
                             dst.with_ref(|p| p),
                             cb)
        })
    }

    pub fn symlink(loop_: &Loop, src: &CString, dst: &CString)
        -> Result<(), UvError>
    {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_symlink(loop_.handle, req,
                                src.with_ref(|p| p),
                                dst.with_ref(|p| p),
                                0, cb)
        })
    }

    pub fn fsync(loop_: &Loop, fd: c_int) -> Result<(), UvError> {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_fsync(loop_.handle, req, fd, cb)
        })
    }

    pub fn datasync(loop_: &Loop, fd: c_int) -> Result<(), UvError> {
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_fdatasync(loop_.handle, req, fd, cb)
        })
    }

    pub fn utime(loop_: &Loop, path: &CString, atime: u64, mtime: u64)
        -> Result<(), UvError>
    {
        // libuv takes seconds
        let atime = atime as libc::c_double / 1000.0;
        let mtime = mtime as libc::c_double / 1000.0;
        execute_nop(|req, cb| unsafe {
            uvll::uv_fs_utime(loop_.handle, req, path.with_ref(|p| p),
                              atime, mtime, cb)
        })
    }

    pub fn get_result(&self) -> c_int {
        unsafe { uvll::get_result_from_fs_req(self.req) }
    }

    pub fn get_stat(&self) -> uvll::uv_stat_t {
        let stat = uvll::uv_stat_t::new();
        unsafe { uvll::populate_stat(self.req, &stat); }
        stat
    }

    pub fn get_ptr(&self) -> *libc::c_void {
        unsafe { uvll::get_ptr_from_fs_req(self.req) }
    }

    pub fn mkstat(&self) -> FileStat {
        let path = unsafe { uvll::get_path_from_fs_req(self.req) };
        let path = unsafe { Path::new(CString::new(path, false)) };
        let stat = self.get_stat();
        fn to_msec(stat: uvll::uv_timespec_t) -> u64 {
            // Be sure to cast to u64 first to prevent overflowing if the tv_sec
            // field is a 32-bit integer.
            (stat.tv_sec as u64) * 1000 + (stat.tv_nsec as u64) / 1000000
        }
        let kind = match (stat.st_mode as c_int) & libc::S_IFMT {
            libc::S_IFREG => io::TypeFile,
            libc::S_IFDIR => io::TypeDirectory,
            libc::S_IFIFO => io::TypeNamedPipe,
            libc::S_IFBLK => io::TypeBlockSpecial,
            libc::S_IFLNK => io::TypeSymlink,
            _ => io::TypeUnknown,
        };
        FileStat {
            path: path,
            size: stat.st_size as u64,
            kind: kind,
            perm: (stat.st_mode as io::FilePermission) & io::AllPermissions,
            created: to_msec(stat.st_birthtim),
            modified: to_msec(stat.st_mtim),
            accessed: to_msec(stat.st_atim),
            unstable: io::UnstableFileStat {
                device: stat.st_dev as u64,
                inode: stat.st_ino as u64,
                rdev: stat.st_rdev as u64,
                nlink: stat.st_nlink as u64,
                uid: stat.st_uid as u64,
                gid: stat.st_gid as u64,
                blksize: stat.st_blksize as u64,
                blocks: stat.st_blocks as u64,
                flags: stat.st_flags as u64,
                gen: stat.st_gen as u64,
            }
        }
    }
}

impl Drop for FsRequest {
    fn drop(&mut self) {
        unsafe {
            if self.fired {
                uvll::uv_fs_req_cleanup(self.req);
            }
            uvll::free_req(self.req);
        }
    }
}

fn execute(f: |*uvll::uv_fs_t, uvll::uv_fs_cb| -> c_int)
    -> Result<FsRequest, UvError>
{
    let mut req = FsRequest {
        fired: false,
        req: unsafe { uvll::malloc_req(uvll::UV_FS) }
    };
    return match f(req.req, fs_cb) {
        0 => {
            req.fired = true;
            let mut slot = None;
            let loop_ = unsafe { uvll::get_loop_from_fs_req(req.req) };
            wait_until_woken_after(&mut slot, &Loop::wrap(loop_), || {
                unsafe { uvll::set_data_for_req(req.req, &slot) }
            });
            match req.get_result() {
                n if n < 0 => Err(UvError(n)),
                _ => Ok(req),
            }
        }
        n => Err(UvError(n))
    };

    extern fn fs_cb(req: *uvll::uv_fs_t) {
        let slot: &mut Option<BlockedTask> = unsafe {
            cast::transmute(uvll::get_data_for_req(req))
        };
        wakeup(slot);
    }
}

fn execute_nop(f: |*uvll::uv_fs_t, uvll::uv_fs_cb| -> c_int)
    -> Result<(), UvError> {
    execute(f).map(|_| {})
}

impl HomingIO for FileWatcher {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle { &mut self.home }
}

impl FileWatcher {
    pub fn new(io: &mut UvIoFactory, fd: c_int,
               close: rtio::CloseBehavior) -> FileWatcher {
        FileWatcher {
            loop_: Loop::wrap(io.uv_loop()),
            fd: fd,
            close: close,
            home: io.make_handle(),
        }
    }

    fn base_read(&mut self, buf: &mut [u8], offset: i64) -> Result<int, IoError> {
        let _m = self.fire_homing_missile();
        let r = FsRequest::read(&self.loop_, self.fd, buf, offset);
        r.map_err(uv_error_to_io_error)
    }
    fn base_write(&mut self, buf: &[u8], offset: i64) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        let r = FsRequest::write(&self.loop_, self.fd, buf, offset);
        r.map_err(uv_error_to_io_error)
    }
    fn seek_common(&mut self, pos: i64, whence: c_int) ->
        Result<u64, IoError>{
        unsafe {
            match libc::lseek(self.fd, pos as libc::off_t, whence) {
                -1 => {
                    Err(IoError {
                        kind: io::OtherIoError,
                        desc: "Failed to lseek.",
                        detail: None
                    })
                },
                n => Ok(n as u64)
            }
        }
    }
}

impl Drop for FileWatcher {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        match self.close {
            rtio::DontClose => {}
            rtio::CloseAsynchronously => {
                unsafe {
                    let req = uvll::malloc_req(uvll::UV_FS);
                    assert_eq!(uvll::uv_fs_close(self.loop_.handle, req,
                                                 self.fd, close_cb), 0);
                }

                extern fn close_cb(req: *uvll::uv_fs_t) {
                    unsafe {
                        uvll::uv_fs_req_cleanup(req);
                        uvll::free_req(req);
                    }
                }
            }
            rtio::CloseSynchronously => {
                let _ = execute_nop(|req, cb| unsafe {
                    uvll::uv_fs_close(self.loop_.handle, req, self.fd, cb)
                });
            }
        }
    }
}

impl rtio::RtioFileStream for FileWatcher {
    fn read(&mut self, buf: &mut [u8]) -> Result<int, IoError> {
        self.base_read(buf, -1)
    }
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        self.base_write(buf, -1)
    }
    fn pread(&mut self, buf: &mut [u8], offset: u64) -> Result<int, IoError> {
        self.base_read(buf, offset as i64)
    }
    fn pwrite(&mut self, buf: &[u8], offset: u64) -> Result<(), IoError> {
        self.base_write(buf, offset as i64)
    }
    fn seek(&mut self, pos: i64, whence: io::SeekStyle) -> Result<u64, IoError> {
        use std::libc::{SEEK_SET, SEEK_CUR, SEEK_END};
        let whence = match whence {
            io::SeekSet => SEEK_SET,
            io::SeekCur => SEEK_CUR,
            io::SeekEnd => SEEK_END
        };
        self.seek_common(pos, whence)
    }
    fn tell(&self) -> Result<u64, IoError> {
        use std::libc::SEEK_CUR;
        // this is temporary
        let self_ = unsafe { cast::transmute_mut(self) };
        self_.seek_common(0, SEEK_CUR)
    }
    fn fsync(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        FsRequest::fsync(&self.loop_, self.fd).map_err(uv_error_to_io_error)
    }
    fn datasync(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        FsRequest::datasync(&self.loop_, self.fd).map_err(uv_error_to_io_error)
    }
    fn truncate(&mut self, offset: i64) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        let r = FsRequest::truncate(&self.loop_, self.fd, offset);
        r.map_err(uv_error_to_io_error)
    }
}

#[cfg(test)]
mod test {
    use std::libc::c_int;
    use std::libc::{O_CREAT, O_RDWR, O_RDONLY, S_IWUSR, S_IRUSR};
    use std::io;
    use std::str;
    use std::slice;
    use super::FsRequest;
    use super::super::Loop;
    use super::super::local_loop;

    fn l() -> &mut Loop { &mut local_loop().loop_ }

    #[test]
    fn file_test_full_simple_sync() {
        let create_flags = O_RDWR | O_CREAT;
        let read_flags = O_RDONLY;
        let mode = S_IWUSR | S_IRUSR;
        let path_str = "./tmp/file_full_simple_sync.txt";

        {
            // open/create
            let result = FsRequest::open(local_loop(), &path_str.to_c_str(),
                                         create_flags as int, mode as int);
            assert!(result.is_ok());
            let result = result.unwrap();
            let fd = result.fd;

            // write
            let result = FsRequest::write(l(), fd, "hello".as_bytes(), -1);
            assert!(result.is_ok());
        }

        {
            // re-open
            let result = FsRequest::open(local_loop(), &path_str.to_c_str(),
                                         read_flags as int, 0);
            assert!(result.is_ok());
            let result = result.unwrap();
            let fd = result.fd;

            // read
            let mut read_mem = slice::from_elem(1000, 0u8);
            let result = FsRequest::read(l(), fd, read_mem, 0);
            assert!(result.is_ok());

            let nread = result.unwrap();
            assert!(nread > 0);
            let read_str = str::from_utf8(read_mem.slice_to(nread as uint)).unwrap();
            assert_eq!(read_str, "hello");
        }
        // unlink
        let result = FsRequest::unlink(l(), &path_str.to_c_str());
        assert!(result.is_ok());
    }

    #[test]
    fn file_test_stat() {
        let path = &"./tmp/file_test_stat_simple".to_c_str();
        let create_flags = (O_RDWR | O_CREAT) as int;
        let mode = (S_IWUSR | S_IRUSR) as int;

        let result = FsRequest::open(local_loop(), path, create_flags, mode);
        assert!(result.is_ok());
        let file = result.unwrap();

        let result = FsRequest::write(l(), file.fd, "hello".as_bytes(), 0);
        assert!(result.is_ok());

        let result = FsRequest::stat(l(), path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().size, 5);

        fn free<T>(_: T) {}
        free(file);

        let result = FsRequest::unlink(l(), path);
        assert!(result.is_ok());
    }

    #[test]
    fn file_test_mk_rm_dir() {
        let path = &"./tmp/mk_rm_dir".to_c_str();
        let mode = S_IWUSR | S_IRUSR;

        let result = FsRequest::mkdir(l(), path, mode);
        assert!(result.is_ok());

        let result = FsRequest::stat(l(), path);
        assert!(result.is_ok());
        assert!(result.unwrap().kind == io::TypeDirectory);

        let result = FsRequest::rmdir(l(), path);
        assert!(result.is_ok());

        let result = FsRequest::stat(l(), path);
        assert!(result.is_err());
    }

    #[test]
    fn file_test_mkdir_chokes_on_double_create() {
        let path = &"./tmp/double_create_dir".to_c_str();
        let mode = S_IWUSR | S_IRUSR;

        let result = FsRequest::stat(l(), path);
        assert!(result.is_err(), "{:?}", result);
        let result = FsRequest::mkdir(l(), path, mode as c_int);
        assert!(result.is_ok(), "{:?}", result);
        let result = FsRequest::mkdir(l(), path, mode as c_int);
        assert!(result.is_err(), "{:?}", result);
        let result = FsRequest::rmdir(l(), path);
        assert!(result.is_ok(), "{:?}", result);
    }

    #[test]
    fn file_test_rmdir_chokes_on_nonexistant_path() {
        let path = &"./tmp/never_existed_dir".to_c_str();
        let result = FsRequest::rmdir(l(), path);
        assert!(result.is_err());
    }
}
