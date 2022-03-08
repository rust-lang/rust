#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use super::err2io;
use crate::io::{self, IoSlice, IoSliceMut, SeekFrom};
use crate::mem;
use crate::net::Shutdown;
use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};

#[derive(Debug)]
pub struct WasiFd {
    fd: OwnedFd,
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
    pub fn datasync(&self) -> io::Result<()> {
        unsafe { wasi::fd_datasync(self.as_raw_fd() as wasi::Fd).map_err(err2io) }
    }

    pub fn pread(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        unsafe { wasi::fd_pread(self.as_raw_fd() as wasi::Fd, iovec(bufs), offset).map_err(err2io) }
    }

    pub fn pwrite(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        unsafe {
            wasi::fd_pwrite(self.as_raw_fd() as wasi::Fd, ciovec(bufs), offset).map_err(err2io)
        }
    }

    pub fn read(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        unsafe { wasi::fd_read(self.as_raw_fd() as wasi::Fd, iovec(bufs)).map_err(err2io) }
    }

    pub fn write(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        unsafe { wasi::fd_write(self.as_raw_fd() as wasi::Fd, ciovec(bufs)).map_err(err2io) }
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, offset) = match pos {
            SeekFrom::Start(pos) => (wasi::WHENCE_SET, pos as i64),
            SeekFrom::End(pos) => (wasi::WHENCE_END, pos),
            SeekFrom::Current(pos) => (wasi::WHENCE_CUR, pos),
        };
        unsafe { wasi::fd_seek(self.as_raw_fd() as wasi::Fd, offset, whence).map_err(err2io) }
    }

    pub fn tell(&self) -> io::Result<u64> {
        unsafe { wasi::fd_tell(self.as_raw_fd() as wasi::Fd).map_err(err2io) }
    }

    // FIXME: __wasi_fd_fdstat_get

    pub fn set_flags(&self, flags: wasi::Fdflags) -> io::Result<()> {
        unsafe { wasi::fd_fdstat_set_flags(self.as_raw_fd() as wasi::Fd, flags).map_err(err2io) }
    }

    pub fn set_rights(&self, base: wasi::Rights, inheriting: wasi::Rights) -> io::Result<()> {
        unsafe {
            wasi::fd_fdstat_set_rights(self.as_raw_fd() as wasi::Fd, base, inheriting)
                .map_err(err2io)
        }
    }

    pub fn sync(&self) -> io::Result<()> {
        unsafe { wasi::fd_sync(self.as_raw_fd() as wasi::Fd).map_err(err2io) }
    }

    pub fn advise(&self, offset: u64, len: u64, advice: wasi::Advice) -> io::Result<()> {
        unsafe {
            wasi::fd_advise(self.as_raw_fd() as wasi::Fd, offset, len, advice).map_err(err2io)
        }
    }

    pub fn allocate(&self, offset: u64, len: u64) -> io::Result<()> {
        unsafe { wasi::fd_allocate(self.as_raw_fd() as wasi::Fd, offset, len).map_err(err2io) }
    }

    pub fn create_directory(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_create_directory(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
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
                self.as_raw_fd() as wasi::Fd,
                old_flags,
                old_path,
                new_fd.as_raw_fd() as wasi::Fd,
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
                self.as_raw_fd() as wasi::Fd,
                dirflags,
                path,
                oflags,
                fs_rights_base,
                fs_rights_inheriting,
                fs_flags,
            )
            .map(|fd| WasiFd::from_raw_fd(fd as RawFd))
            .map_err(err2io)
        }
    }

    pub fn readdir(&self, buf: &mut [u8], cookie: wasi::Dircookie) -> io::Result<usize> {
        unsafe {
            wasi::fd_readdir(self.as_raw_fd() as wasi::Fd, buf.as_mut_ptr(), buf.len(), cookie)
                .map_err(err2io)
        }
    }

    pub fn readlink(&self, path: &str, buf: &mut [u8]) -> io::Result<usize> {
        unsafe {
            wasi::path_readlink(self.as_raw_fd() as wasi::Fd, path, buf.as_mut_ptr(), buf.len())
                .map_err(err2io)
        }
    }

    pub fn rename(&self, old_path: &str, new_fd: &WasiFd, new_path: &str) -> io::Result<()> {
        unsafe {
            wasi::path_rename(
                self.as_raw_fd() as wasi::Fd,
                old_path,
                new_fd.as_raw_fd() as wasi::Fd,
                new_path,
            )
            .map_err(err2io)
        }
    }

    pub fn filestat_get(&self) -> io::Result<wasi::Filestat> {
        unsafe { wasi::fd_filestat_get(self.as_raw_fd() as wasi::Fd).map_err(err2io) }
    }

    pub fn filestat_set_times(
        &self,
        atim: wasi::Timestamp,
        mtim: wasi::Timestamp,
        fstflags: wasi::Fstflags,
    ) -> io::Result<()> {
        unsafe {
            wasi::fd_filestat_set_times(self.as_raw_fd() as wasi::Fd, atim, mtim, fstflags)
                .map_err(err2io)
        }
    }

    pub fn filestat_set_size(&self, size: u64) -> io::Result<()> {
        unsafe { wasi::fd_filestat_set_size(self.as_raw_fd() as wasi::Fd, size).map_err(err2io) }
    }

    pub fn path_filestat_get(
        &self,
        flags: wasi::Lookupflags,
        path: &str,
    ) -> io::Result<wasi::Filestat> {
        unsafe {
            wasi::path_filestat_get(self.as_raw_fd() as wasi::Fd, flags, path).map_err(err2io)
        }
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
            wasi::path_filestat_set_times(
                self.as_raw_fd() as wasi::Fd,
                flags,
                path,
                atim,
                mtim,
                fstflags,
            )
            .map_err(err2io)
        }
    }

    pub fn symlink(&self, old_path: &str, new_path: &str) -> io::Result<()> {
        unsafe {
            wasi::path_symlink(old_path, self.as_raw_fd() as wasi::Fd, new_path).map_err(err2io)
        }
    }

    pub fn unlink_file(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_unlink_file(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
    }

    pub fn remove_directory(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_remove_directory(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
    }

    pub fn sock_accept(&self, flags: wasi::Fdflags) -> io::Result<wasi::Fd> {
        unsafe { wasi::sock_accept(self.as_raw_fd() as wasi::Fd, flags).map_err(err2io) }
    }

    pub fn sock_recv(
        &self,
        ri_data: &mut [IoSliceMut<'_>],
        ri_flags: wasi::Riflags,
    ) -> io::Result<(usize, wasi::Roflags)> {
        unsafe {
            wasi::sock_recv(self.as_raw_fd() as wasi::Fd, iovec(ri_data), ri_flags).map_err(err2io)
        }
    }

    pub fn sock_send(&self, si_data: &[IoSlice<'_>], si_flags: wasi::Siflags) -> io::Result<usize> {
        unsafe {
            wasi::sock_send(self.as_raw_fd() as wasi::Fd, ciovec(si_data), si_flags).map_err(err2io)
        }
    }

    pub fn sock_shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Read => wasi::SDFLAGS_RD,
            Shutdown::Write => wasi::SDFLAGS_WR,
            Shutdown::Both => wasi::SDFLAGS_WR | wasi::SDFLAGS_RD,
        };
        unsafe { wasi::sock_shutdown(self.as_raw_fd() as wasi::Fd, how).map_err(err2io) }
    }
}

impl AsInner<OwnedFd> for WasiFd {
    fn as_inner(&self) -> &OwnedFd {
        &self.fd
    }
}

impl AsInnerMut<OwnedFd> for WasiFd {
    fn as_inner_mut(&mut self) -> &mut OwnedFd {
        &mut self.fd
    }
}

impl IntoInner<OwnedFd> for WasiFd {
    fn into_inner(self) -> OwnedFd {
        self.fd
    }
}

impl FromInner<OwnedFd> for WasiFd {
    fn from_inner(owned_fd: OwnedFd) -> Self {
        Self { fd: owned_fd }
    }
}

impl AsFd for WasiFd {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl AsRawFd for WasiFd {
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

impl IntoRawFd for WasiFd {
    fn into_raw_fd(self) -> RawFd {
        self.fd.into_raw_fd()
    }
}

impl FromRawFd for WasiFd {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self { fd: FromRawFd::from_raw_fd(raw_fd) } }
    }
}
