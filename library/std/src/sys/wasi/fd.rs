#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use super::err2io;
use crate::io::{self, IoSlice, IoSliceMut, SeekFrom};
use crate::mem;
use crate::net::Shutdown;
use crate::os::raw::c_int;

#[derive(Debug)]
pub struct WasiFd {
    fd: c_int,
}

fn iovec<'a>(a: &'a mut [IoSliceMut<'_>]) -> &'a [wasi::Iovec] {
    assert_eq!(mem::size_of::<IoSliceMut<'_>>(), mem::size_of::<wasi::Iovec>());
    assert_eq!(mem::align_of::<IoSliceMut<'_>>(), mem::align_of::<wasi::Iovec>());
    // SAFETY: `IoSliceMut` and `IoVec` have exactly the same memory layout
    unsafe { mem::transmute(a) }
}

fn ciovec<'a>(a: &'a [IoSlice<'_>]) -> &'a [wasi::Ciovec] {
    assert_eq!(mem::size_of::<IoSlice<'_>>(), mem::size_of::<wasi::Ciovec>());
    assert_eq!(mem::align_of::<IoSlice<'_>>(), mem::align_of::<wasi::Ciovec>());
    // SAFETY: `IoSlice` and `CIoVec` have exactly the same memory layout
    unsafe { mem::transmute(a) }
}

impl WasiFd {
    pub unsafe fn from_raw(fd: c_int) -> WasiFd {
        WasiFd { fd }
    }

    pub fn into_raw(self) -> c_int {
        let ret = self.fd;
        mem::forget(self);
        ret
    }

    pub fn as_raw(&self) -> c_int {
        self.fd
    }

    pub fn datasync(&self) -> io::Result<()> {
        unsafe { wasi::fd_datasync(self.fd as wasi::Fd).map_err(err2io) }
    }

    pub fn pread(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        unsafe { wasi::fd_pread(self.fd as wasi::Fd, iovec(bufs), offset).map_err(err2io) }
    }

    pub fn pwrite(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        unsafe { wasi::fd_pwrite(self.fd as wasi::Fd, ciovec(bufs), offset).map_err(err2io) }
    }

    pub fn read(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        unsafe { wasi::fd_read(self.fd as wasi::Fd, iovec(bufs)).map_err(err2io) }
    }

    pub fn write(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        unsafe { wasi::fd_write(self.fd as wasi::Fd, ciovec(bufs)).map_err(err2io) }
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, offset) = match pos {
            SeekFrom::Start(pos) => (wasi::WHENCE_SET, pos as i64),
            SeekFrom::End(pos) => (wasi::WHENCE_END, pos),
            SeekFrom::Current(pos) => (wasi::WHENCE_CUR, pos),
        };
        unsafe { wasi::fd_seek(self.fd as wasi::Fd, offset, whence).map_err(err2io) }
    }

    pub fn tell(&self) -> io::Result<u64> {
        unsafe { wasi::fd_tell(self.fd as wasi::Fd).map_err(err2io) }
    }

    // FIXME: __wasi_fd_fdstat_get

    pub fn set_flags(&self, flags: wasi::Fdflags) -> io::Result<()> {
        unsafe { wasi::fd_fdstat_set_flags(self.fd as wasi::Fd, flags).map_err(err2io) }
    }

    pub fn set_rights(&self, base: wasi::Rights, inheriting: wasi::Rights) -> io::Result<()> {
        unsafe { wasi::fd_fdstat_set_rights(self.fd as wasi::Fd, base, inheriting).map_err(err2io) }
    }

    pub fn sync(&self) -> io::Result<()> {
        unsafe { wasi::fd_sync(self.fd as wasi::Fd).map_err(err2io) }
    }

    pub fn advise(&self, offset: u64, len: u64, advice: wasi::Advice) -> io::Result<()> {
        unsafe { wasi::fd_advise(self.fd as wasi::Fd, offset, len, advice).map_err(err2io) }
    }

    pub fn allocate(&self, offset: u64, len: u64) -> io::Result<()> {
        unsafe { wasi::fd_allocate(self.fd as wasi::Fd, offset, len).map_err(err2io) }
    }

    pub fn create_directory(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_create_directory(self.fd as wasi::Fd, path).map_err(err2io) }
    }

    pub fn link(
        &self,
        old_flags: wasi::Lookupflags,
        old_path: &str,
        new_fd: &WasiFd,
        new_path: &str,
    ) -> io::Result<()> {
        unsafe {
            wasi::path_link(
                self.fd as wasi::Fd,
                old_flags,
                old_path,
                new_fd.fd as wasi::Fd,
                new_path,
            )
            .map_err(err2io)
        }
    }

    pub fn open(
        &self,
        dirflags: wasi::Lookupflags,
        path: &str,
        oflags: wasi::Oflags,
        fs_rights_base: wasi::Rights,
        fs_rights_inheriting: wasi::Rights,
        fs_flags: wasi::Fdflags,
    ) -> io::Result<WasiFd> {
        unsafe {
            wasi::path_open(
                self.fd as wasi::Fd,
                dirflags,
                path,
                oflags,
                fs_rights_base,
                fs_rights_inheriting,
                fs_flags,
            )
            .map(|fd| WasiFd::from_raw(fd as c_int))
            .map_err(err2io)
        }
    }

    pub fn readdir(&self, buf: &mut [u8], cookie: wasi::Dircookie) -> io::Result<usize> {
        unsafe {
            wasi::fd_readdir(self.fd as wasi::Fd, buf.as_mut_ptr(), buf.len(), cookie)
                .map_err(err2io)
        }
    }

    pub fn readlink(&self, path: &str, buf: &mut [u8]) -> io::Result<usize> {
        unsafe {
            wasi::path_readlink(self.fd as wasi::Fd, path, buf.as_mut_ptr(), buf.len())
                .map_err(err2io)
        }
    }

    pub fn rename(&self, old_path: &str, new_fd: &WasiFd, new_path: &str) -> io::Result<()> {
        unsafe {
            wasi::path_rename(self.fd as wasi::Fd, old_path, new_fd.fd as wasi::Fd, new_path)
                .map_err(err2io)
        }
    }

    pub fn filestat_get(&self) -> io::Result<wasi::Filestat> {
        unsafe { wasi::fd_filestat_get(self.fd as wasi::Fd).map_err(err2io) }
    }

    pub fn filestat_set_times(
        &self,
        atim: wasi::Timestamp,
        mtim: wasi::Timestamp,
        fstflags: wasi::Fstflags,
    ) -> io::Result<()> {
        unsafe {
            wasi::fd_filestat_set_times(self.fd as wasi::Fd, atim, mtim, fstflags).map_err(err2io)
        }
    }

    pub fn filestat_set_size(&self, size: u64) -> io::Result<()> {
        unsafe { wasi::fd_filestat_set_size(self.fd as wasi::Fd, size).map_err(err2io) }
    }

    pub fn path_filestat_get(
        &self,
        flags: wasi::Lookupflags,
        path: &str,
    ) -> io::Result<wasi::Filestat> {
        unsafe { wasi::path_filestat_get(self.fd as wasi::Fd, flags, path).map_err(err2io) }
    }

    pub fn path_filestat_set_times(
        &self,
        flags: wasi::Lookupflags,
        path: &str,
        atim: wasi::Timestamp,
        mtim: wasi::Timestamp,
        fstflags: wasi::Fstflags,
    ) -> io::Result<()> {
        unsafe {
            wasi::path_filestat_set_times(self.fd as wasi::Fd, flags, path, atim, mtim, fstflags)
                .map_err(err2io)
        }
    }

    pub fn symlink(&self, old_path: &str, new_path: &str) -> io::Result<()> {
        unsafe { wasi::path_symlink(old_path, self.fd as wasi::Fd, new_path).map_err(err2io) }
    }

    pub fn unlink_file(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_unlink_file(self.fd as wasi::Fd, path).map_err(err2io) }
    }

    pub fn remove_directory(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_remove_directory(self.fd as wasi::Fd, path).map_err(err2io) }
    }

    pub fn sock_recv(
        &self,
        ri_data: &mut [IoSliceMut<'_>],
        ri_flags: wasi::Riflags,
    ) -> io::Result<(usize, wasi::Roflags)> {
        unsafe { wasi::sock_recv(self.fd as wasi::Fd, iovec(ri_data), ri_flags).map_err(err2io) }
    }

    pub fn sock_send(&self, si_data: &[IoSlice<'_>], si_flags: wasi::Siflags) -> io::Result<usize> {
        unsafe { wasi::sock_send(self.fd as wasi::Fd, ciovec(si_data), si_flags).map_err(err2io) }
    }

    pub fn sock_shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Read => wasi::SDFLAGS_RD,
            Shutdown::Write => wasi::SDFLAGS_WR,
            Shutdown::Both => wasi::SDFLAGS_WR | wasi::SDFLAGS_RD,
        };
        unsafe { wasi::sock_shutdown(self.fd as wasi::Fd, how).map_err(err2io) }
    }
}

impl Drop for WasiFd {
    fn drop(&mut self) {
        // FIXME: can we handle the return code here even though we can't on
        // unix?
        let _ = unsafe { wasi::fd_close(self.fd as wasi::Fd) };
    }
}
