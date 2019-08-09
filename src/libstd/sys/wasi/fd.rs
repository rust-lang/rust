#![allow(dead_code)]

use crate::io::{self, IoSlice, IoSliceMut, SeekFrom};
use crate::mem;
use crate::net::Shutdown;
use crate::sys::cvt_wasi;
use libc::{self, c_char, c_void};

#[derive(Debug)]
pub struct WasiFd {
    fd: libc::__wasi_fd_t,
}

// FIXME: these should probably all be fancier structs, builders, enums, etc
pub type LookupFlags = u32;
pub type FdFlags = u16;
pub type Advice = u8;
pub type Rights = u64;
pub type Oflags = u16;
pub type DirCookie = u64;
pub type Timestamp = u64;
pub type FstFlags = u16;
pub type RiFlags = u16;
pub type RoFlags = u16;
pub type SiFlags = u16;

fn iovec(a: &mut [IoSliceMut<'_>]) -> (*const libc::__wasi_iovec_t, usize) {
    assert_eq!(
        mem::size_of::<IoSliceMut<'_>>(),
        mem::size_of::<libc::__wasi_iovec_t>()
    );
    assert_eq!(
        mem::align_of::<IoSliceMut<'_>>(),
        mem::align_of::<libc::__wasi_iovec_t>()
    );
    (a.as_ptr() as *const libc::__wasi_iovec_t, a.len())
}

fn ciovec(a: &[IoSlice<'_>]) -> (*const libc::__wasi_ciovec_t, usize) {
    assert_eq!(
        mem::size_of::<IoSlice<'_>>(),
        mem::size_of::<libc::__wasi_ciovec_t>()
    );
    assert_eq!(
        mem::align_of::<IoSlice<'_>>(),
        mem::align_of::<libc::__wasi_ciovec_t>()
    );
    (a.as_ptr() as *const libc::__wasi_ciovec_t, a.len())
}

impl WasiFd {
    pub unsafe fn from_raw(fd: libc::__wasi_fd_t) -> WasiFd {
        WasiFd { fd }
    }

    pub fn into_raw(self) -> libc::__wasi_fd_t {
        let ret = self.fd;
        mem::forget(self);
        ret
    }

    pub fn as_raw(&self) -> libc::__wasi_fd_t {
        self.fd
    }

    pub fn datasync(&self) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_datasync(self.fd) })
    }

    pub fn pread(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        let mut read = 0;
        let (ptr, len) = iovec(bufs);
        cvt_wasi(unsafe { libc::__wasi_fd_pread(self.fd, ptr, len, offset, &mut read) })?;
        Ok(read)
    }

    pub fn pwrite(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        let mut read = 0;
        let (ptr, len) = ciovec(bufs);
        cvt_wasi(unsafe { libc::__wasi_fd_pwrite(self.fd, ptr, len, offset, &mut read) })?;
        Ok(read)
    }

    pub fn read(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let mut read = 0;
        let (ptr, len) = iovec(bufs);
        cvt_wasi(unsafe { libc::__wasi_fd_read(self.fd, ptr, len, &mut read) })?;
        Ok(read)
    }

    pub fn write(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let mut read = 0;
        let (ptr, len) = ciovec(bufs);
        cvt_wasi(unsafe { libc::__wasi_fd_write(self.fd, ptr, len, &mut read) })?;
        Ok(read)
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, offset) = match pos {
            SeekFrom::Start(pos) => (libc::__WASI_WHENCE_SET, pos as i64),
            SeekFrom::End(pos) => (libc::__WASI_WHENCE_END, pos),
            SeekFrom::Current(pos) => (libc::__WASI_WHENCE_CUR, pos),
        };
        let mut pos = 0;
        cvt_wasi(unsafe { libc::__wasi_fd_seek(self.fd, offset, whence, &mut pos) })?;
        Ok(pos)
    }

    pub fn tell(&self) -> io::Result<u64> {
        let mut pos = 0;
        cvt_wasi(unsafe { libc::__wasi_fd_tell(self.fd, &mut pos) })?;
        Ok(pos)
    }

    // FIXME: __wasi_fd_fdstat_get

    pub fn set_flags(&self, flags: FdFlags) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_fdstat_set_flags(self.fd, flags) })
    }

    pub fn set_rights(&self, base: Rights, inheriting: Rights) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_fdstat_set_rights(self.fd, base, inheriting) })
    }

    pub fn sync(&self) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_sync(self.fd) })
    }

    pub fn advise(&self, offset: u64, len: u64, advice: Advice) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_advise(self.fd, offset, len, advice as u8) })
    }

    pub fn allocate(&self, offset: u64, len: u64) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_allocate(self.fd, offset, len) })
    }

    pub fn create_directory(&self, path: &[u8]) -> io::Result<()> {
        cvt_wasi(unsafe {
            libc::__wasi_path_create_directory(self.fd, path.as_ptr() as *const c_char, path.len())
        })
    }

    pub fn link(
        &self,
        old_flags: LookupFlags,
        old_path: &[u8],
        new_fd: &WasiFd,
        new_path: &[u8],
    ) -> io::Result<()> {
        cvt_wasi(unsafe {
            libc::__wasi_path_link(
                self.fd,
                old_flags,
                old_path.as_ptr() as *const c_char,
                old_path.len(),
                new_fd.fd,
                new_path.as_ptr() as *const c_char,
                new_path.len(),
            )
        })
    }

    pub fn open(
        &self,
        dirflags: LookupFlags,
        path: &[u8],
        oflags: Oflags,
        fs_rights_base: Rights,
        fs_rights_inheriting: Rights,
        fs_flags: FdFlags,
    ) -> io::Result<WasiFd> {
        unsafe {
            let mut fd = 0;
            cvt_wasi(libc::__wasi_path_open(
                self.fd,
                dirflags,
                path.as_ptr() as *const c_char,
                path.len(),
                oflags,
                fs_rights_base,
                fs_rights_inheriting,
                fs_flags,
                &mut fd,
            ))?;
            Ok(WasiFd::from_raw(fd))
        }
    }

    pub fn readdir(&self, buf: &mut [u8], cookie: DirCookie) -> io::Result<usize> {
        let mut used = 0;
        cvt_wasi(unsafe {
            libc::__wasi_fd_readdir(
                self.fd,
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
                cookie,
                &mut used,
            )
        })?;
        Ok(used)
    }

    pub fn readlink(&self, path: &[u8], buf: &mut [u8]) -> io::Result<usize> {
        let mut used = 0;
        cvt_wasi(unsafe {
            libc::__wasi_path_readlink(
                self.fd,
                path.as_ptr() as *const c_char,
                path.len(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
                &mut used,
            )
        })?;
        Ok(used)
    }

    pub fn rename(&self, old_path: &[u8], new_fd: &WasiFd, new_path: &[u8]) -> io::Result<()> {
        cvt_wasi(unsafe {
            libc::__wasi_path_rename(
                self.fd,
                old_path.as_ptr() as *const c_char,
                old_path.len(),
                new_fd.fd,
                new_path.as_ptr() as *const c_char,
                new_path.len(),
            )
        })
    }

    pub fn filestat_get(&self, buf: *mut libc::__wasi_filestat_t) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_filestat_get(self.fd, buf) })
    }

    pub fn filestat_set_times(
        &self,
        atim: Timestamp,
        mtim: Timestamp,
        fstflags: FstFlags,
    ) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_filestat_set_times(self.fd, atim, mtim, fstflags) })
    }

    pub fn filestat_set_size(&self, size: u64) -> io::Result<()> {
        cvt_wasi(unsafe { libc::__wasi_fd_filestat_set_size(self.fd, size) })
    }

    pub fn path_filestat_get(
        &self,
        flags: LookupFlags,
        path: &[u8],
        buf: *mut libc::__wasi_filestat_t,
    ) -> io::Result<()> {
        cvt_wasi(unsafe {
            libc::__wasi_path_filestat_get(
                self.fd,
                flags,
                path.as_ptr() as *const c_char,
                path.len(),
                buf,
            )
        })
    }

    pub fn path_filestat_set_times(
        &self,
        flags: LookupFlags,
        path: &[u8],
        atim: Timestamp,
        mtim: Timestamp,
        fstflags: FstFlags,
    ) -> io::Result<()> {
        cvt_wasi(unsafe {
            libc::__wasi_path_filestat_set_times(
                self.fd,
                flags,
                path.as_ptr() as *const c_char,
                path.len(),
                atim,
                mtim,
                fstflags,
            )
        })
    }

    pub fn symlink(&self, old_path: &[u8], new_path: &[u8]) -> io::Result<()> {
        cvt_wasi(unsafe {
            libc::__wasi_path_symlink(
                old_path.as_ptr() as *const c_char,
                old_path.len(),
                self.fd,
                new_path.as_ptr() as *const c_char,
                new_path.len(),
            )
        })
    }

    pub fn unlink_file(&self, path: &[u8]) -> io::Result<()> {
        cvt_wasi(unsafe {
            libc::__wasi_path_unlink_file(self.fd, path.as_ptr() as *const c_char, path.len())
        })
    }

    pub fn remove_directory(&self, path: &[u8]) -> io::Result<()> {
        cvt_wasi(unsafe {
            libc::__wasi_path_remove_directory(self.fd, path.as_ptr() as *const c_char, path.len())
        })
    }

    pub fn sock_recv(
        &self,
        ri_data: &mut [IoSliceMut<'_>],
        ri_flags: RiFlags,
    ) -> io::Result<(usize, RoFlags)> {
        let mut ro_datalen = 0;
        let mut ro_flags = 0;
        let (ptr, len) = iovec(ri_data);
        cvt_wasi(unsafe {
            libc::__wasi_sock_recv(self.fd, ptr, len, ri_flags, &mut ro_datalen, &mut ro_flags)
        })?;
        Ok((ro_datalen, ro_flags))
    }

    pub fn sock_send(&self, si_data: &[IoSlice<'_>], si_flags: SiFlags) -> io::Result<usize> {
        let mut so_datalen = 0;
        let (ptr, len) = ciovec(si_data);
        cvt_wasi(unsafe { libc::__wasi_sock_send(self.fd, ptr, len, si_flags, &mut so_datalen) })?;
        Ok(so_datalen)
    }

    pub fn sock_shutdown(&self, how: Shutdown) -> io::Result<()> {
        let how = match how {
            Shutdown::Read => libc::__WASI_SHUT_RD,
            Shutdown::Write => libc::__WASI_SHUT_WR,
            Shutdown::Both => libc::__WASI_SHUT_WR | libc::__WASI_SHUT_RD,
        };
        cvt_wasi(unsafe { libc::__wasi_sock_shutdown(self.fd, how) })?;
        Ok(())
    }
}

impl Drop for WasiFd {
    fn drop(&mut self) {
        unsafe {
            // FIXME: can we handle the return code here even though we can't on
            // unix?
            libc::__wasi_fd_close(self.fd);
        }
    }
}
