#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use super::err2io;
use crate::io::{self, IoSlice, IoSliceMut, SeekFrom};
use crate::mem;
use crate::net::Shutdown;
use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};
use crate::io::Read;

#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct WasiFd {
    pub(super) fd: OwnedFd,
}

pub use WasiFd as FileDesc;

pub(super) fn iovec<'a>(a: &'a mut [IoSliceMut<'_>]) -> &'a [wasi::Iovec] {
    assert_eq!(mem::size_of::<IoSliceMut<'_>>(), mem::size_of::<wasi::Iovec>());
    assert_eq!(mem::align_of::<IoSliceMut<'_>>(), mem::align_of::<wasi::Iovec>());
    // SAFETY: `IoSliceMut` and `IoVec` have exactly the same memory layout
    unsafe { mem::transmute(a) }
}

pub(super) fn ciovec<'a>(a: &'a [IoSlice<'_>]) -> &'a [wasi::Ciovec] {
    assert_eq!(mem::size_of::<IoSlice<'_>>(), mem::size_of::<wasi::Ciovec>());
    assert_eq!(mem::align_of::<IoSlice<'_>>(), mem::align_of::<wasi::Ciovec>());
    // SAFETY: `IoSlice` and `CIoVec` have exactly the same memory layout
    unsafe { mem::transmute(a) }
}

impl WasiFd {
    pub(crate) fn datasync(&self) -> io::Result<()> {
        unsafe { wasi::fd_datasync(self.as_raw_fd() as wasi::Fd).map_err(err2io) }
    }

    pub(crate) fn pread(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        unsafe { wasi::fd_pread(self.as_raw_fd() as wasi::Fd, iovec(bufs), offset).map_err(err2io) }
    }

    pub(crate) fn pwrite(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        unsafe {
            wasi::fd_pwrite(self.as_raw_fd() as wasi::Fd, ciovec(bufs), offset).map_err(err2io)
        }
    }

    pub(crate) fn read(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        unsafe { wasi::fd_read(self.as_raw_fd() as wasi::Fd, iovec(bufs)).map_err(err2io) }
    }

    pub(crate) fn write(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        unsafe { wasi::fd_write(self.as_raw_fd() as wasi::Fd, ciovec(bufs)).map_err(err2io) }
    }

    pub(crate) fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, offset) = match pos {
            SeekFrom::Start(pos) => (wasi::WHENCE_SET, pos as i64),
            SeekFrom::End(pos) => (wasi::WHENCE_END, pos),
            SeekFrom::Current(pos) => (wasi::WHENCE_CUR, pos),
        };
        unsafe { wasi::fd_seek(self.as_raw_fd() as wasi::Fd, offset, whence).map_err(err2io) }
    }

    pub(crate) fn tell(&self) -> io::Result<u64> {
        unsafe { wasi::fd_tell(self.as_raw_fd() as wasi::Fd).map_err(err2io) }
    }

    // FIXME: __wasi_fd_fdstat_get

    pub(crate) fn set_flags(&self, flags: wasi::Fdflags) -> io::Result<()> {
        unsafe { wasi::fd_fdstat_set_flags(self.as_raw_fd() as wasi::Fd, flags).map_err(err2io) }
    }

    pub(crate) fn set_rights(&self, base: wasi::Rights, inheriting: wasi::Rights) -> io::Result<()> {
        unsafe {
            wasi::fd_fdstat_set_rights(self.as_raw_fd() as wasi::Fd, base, inheriting)
                .map_err(err2io)
        }
    }

    pub(crate) fn sync(&self) -> io::Result<()> {
        unsafe { wasi::fd_sync(self.as_raw_fd() as wasi::Fd).map_err(err2io) }
    }

    pub(crate) fn advise(&self, offset: u64, len: u64, advice: wasi::Advice) -> io::Result<()> {
        unsafe {
            wasi::fd_advise(self.as_raw_fd() as wasi::Fd, offset, len, advice).map_err(err2io)
        }
    }

    pub(crate) fn allocate(&self, offset: u64, len: u64) -> io::Result<()> {
        unsafe { wasi::fd_allocate(self.as_raw_fd() as wasi::Fd, offset, len).map_err(err2io) }
    }

    pub(crate) fn create_directory(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_create_directory(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
    }

    pub(crate) fn link(
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

    pub(crate) fn open(
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

    pub(crate) fn readdir(&self, buf: &mut [u8], cookie: wasi::Dircookie) -> io::Result<usize> {
        unsafe {
            wasi::fd_readdir(self.as_raw_fd() as wasi::Fd, buf.as_mut_ptr(), buf.len(), cookie)
                .map_err(err2io)
        }
    }

    pub(crate) fn readlink(&self, path: &str, buf: &mut [u8]) -> io::Result<usize> {
        unsafe {
            wasi::path_readlink(self.as_raw_fd() as wasi::Fd, path, buf.as_mut_ptr(), buf.len())
                .map_err(err2io)
        }
    }

    pub(crate) fn rename(&self, old_path: &str, new_fd: &WasiFd, new_path: &str) -> io::Result<()> {
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

    pub(crate) fn filestat_get(&self) -> io::Result<wasi::Filestat> {
        unsafe { wasi::fd_filestat_get(self.as_raw_fd() as wasi::Fd).map_err(err2io) }
    }

    pub(crate) fn filestat_set_times(
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

    pub(crate) fn filestat_set_size(&self, size: u64) -> io::Result<()> {
        unsafe { wasi::fd_filestat_set_size(self.as_raw_fd() as wasi::Fd, size).map_err(err2io) }
    }

    pub(crate) fn path_filestat_get(
        &self,
        flags: wasi::Lookupflags,
        path: &str,
    ) -> io::Result<wasi::Filestat> {
        unsafe {
            wasi::path_filestat_get(self.as_raw_fd() as wasi::Fd, flags, path).map_err(err2io)
        }
    }

    pub(crate) fn path_filestat_set_times(
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

    pub(crate) fn symlink(&self, old_path: &str, new_path: &str) -> io::Result<()> {
        unsafe {
            wasi::path_symlink(old_path, self.as_raw_fd() as wasi::Fd, new_path).map_err(err2io)
        }
    }

    pub(crate) fn unlink_file(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_unlink_file(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
    }

    pub(crate) fn remove_directory(&self, path: &str) -> io::Result<()> {
        unsafe { wasi::path_remove_directory(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
    }

    pub(crate) fn sock_accept(&self, flags: wasi::Fdflags) -> io::Result<wasi::Fd> {
        let ret = unsafe {
            wasi::sock_accept_v2(self.as_raw_fd() as wasi::Fd, flags).map_err(err2io)?
        };
        Ok(ret.0)
    }

    pub(crate) fn sock_recv(
        &self,
        ri_data: &mut [IoSliceMut<'_>],
        ri_flags: wasi::Riflags,
    ) -> io::Result<(usize, wasi::Roflags)> {
        let (amt, flags) = unsafe {
            wasi::sock_recv(self.as_raw_fd() as wasi::Fd, iovec(ri_data), ri_flags).map_err(err2io)?
        };
        Ok((amt as usize, flags))
    }

    pub(crate) fn sock_send(&self, si_data: &[IoSlice<'_>], si_flags: wasi::Siflags) -> io::Result<usize> {
        unsafe {
            wasi::sock_send(self.as_raw_fd() as wasi::Fd, ciovec(si_data), si_flags).map(|a| a as usize).map_err(err2io)
        }
    }

    pub(crate) fn sock_shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Read => wasi::SDFLAGS_RD,
            Shutdown::Write => wasi::SDFLAGS_WR,
            Shutdown::Both => wasi::SDFLAGS_WR | wasi::SDFLAGS_RD,
        };
        unsafe { wasi::sock_shutdown(self.as_raw_fd() as wasi::Fd, how).map_err(err2io) }
    }
    
    pub(crate) fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        let fdstat = unsafe {
            wasi::fd_fdstat_get(self.as_raw_fd() as wasi::Fd).map_err(err2io)?
        };

        let mut flags = fdstat.fs_flags;

        if nonblocking {
            flags |= wasi::FDFLAGS_NONBLOCK;
        } else {
            flags &= !wasi::FDFLAGS_NONBLOCK;
        }

        unsafe {
            wasi::fd_fdstat_set_flags(self.as_raw_fd() as wasi::Fd, flags)
                .map_err(err2io)?;
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn duplicate(&self) -> io::Result<Self> {
        let raw_fd = unsafe {
            wasi::fd_dup(self.as_raw_fd() as wasi::Fd).map_err(err2io)?
        } as RawFd;
        Ok(
            unsafe { Self { fd: OwnedFd::from_raw_fd(raw_fd) } }
        )
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Read for &'a WasiFd {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        WasiFd::read(self, &mut [IoSliceMut::new(buf)])
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsInner<OwnedFd> for WasiFd {
    fn as_inner(&self) -> &OwnedFd {
        &self.fd
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsInnerMut<OwnedFd> for WasiFd {
    fn as_inner_mut(&mut self) -> &mut OwnedFd {
        &mut self.fd
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl IntoInner<OwnedFd> for WasiFd {
    fn into_inner(self) -> OwnedFd {
        self.fd
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromInner<OwnedFd> for WasiFd {
    fn from_inner(owned_fd: OwnedFd) -> Self {
        Self { fd: owned_fd }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsFd for WasiFd {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawFd for WasiFd {
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl IntoRawFd for WasiFd {
    fn into_raw_fd(self) -> RawFd {
        self.fd.into_raw_fd()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromRawFd for WasiFd {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self { fd: OwnedFd::from_raw_fd(raw_fd) } }
    }
}
