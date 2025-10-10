//! WASIp1-specific extensions to primitives in the [`std::fs`] module.
//!
//! [`std::fs`]: crate::fs

#![unstable(feature = "wasi_ext", issue = "71213")]

// Used for `File::read` on intra-doc links
#[allow(unused_imports)]
use io::{Read, Write};

#[cfg(target_env = "p1")]
use crate::ffi::OsStr;
use crate::fs::{self, File, OpenOptions};
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
#[cfg(target_env = "p1")]
use crate::os::fd::AsRawFd;
use crate::path::Path;
#[cfg(target_env = "p1")]
use crate::sys::err2io;
use crate::sys_common::{AsInner, AsInnerMut};

/// WASI-specific extensions to [`File`].
pub trait FileExt {
    /// Reads a number of bytes starting from a given offset.
    ///
    /// Returns the number of bytes read.
    ///
    /// The offset is relative to the start of the file and thus independent
    /// from the current cursor.
    ///
    /// The current file cursor is not affected by this function.
    ///
    /// Note that similar to [`File::read`], it is not an error to return with a
    /// short read.
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;

    /// Reads a number of bytes starting from a given offset.
    ///
    /// Returns the number of bytes read.
    ///
    /// The offset is relative to the start of the file and thus independent
    /// from the current cursor.
    ///
    /// The current file cursor is not affected by this function.
    ///
    /// Note that similar to [`File::read_vectored`], it is not an error to
    /// return with a short read.
    fn read_vectored_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize>;

    /// Reads some bytes starting from a given offset into the buffer.
    ///
    /// This equivalent to the [`read_at`](FileExt::read_at) method, except that it is passed a
    /// [`BorrowedCursor`] rather than `&mut [u8]` to allow use with uninitialized buffers. The new
    /// data will be appended to any existing contents of `buf`.
    fn read_buf_at(&self, buf: BorrowedCursor<'_>, offset: u64) -> io::Result<()>;

    /// Reads the exact number of byte required to fill `buf` from the given offset.
    ///
    /// The offset is relative to the start of the file and thus independent
    /// from the current cursor.
    ///
    /// The current file cursor is not affected by this function.
    ///
    /// Similar to [`Read::read_exact`] but uses [`read_at`] instead of `read`.
    ///
    /// [`read_at`]: FileExt::read_at
    ///
    /// # Errors
    ///
    /// If this function encounters an error of the kind
    /// [`io::ErrorKind::Interrupted`] then the error is ignored and the operation
    /// will continue.
    ///
    /// If this function encounters an "end of file" before completely filling
    /// the buffer, it returns an error of the kind [`io::ErrorKind::UnexpectedEof`].
    /// The contents of `buf` are unspecified in this case.
    ///
    /// If any other read error is encountered then this function immediately
    /// returns. The contents of `buf` are unspecified in this case.
    ///
    /// If this function returns an error, it is unspecified how many bytes it
    /// has read, but it will never read more than would be necessary to
    /// completely fill the buffer.
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.read_at(buf, offset) {
                Ok(0) => break,
                Ok(n) => {
                    let tmp = buf;
                    buf = &mut tmp[n..];
                    offset += n as u64;
                }
                Err(ref e) if e.is_interrupted() => {}
                Err(e) => return Err(e),
            }
        }
        if !buf.is_empty() { Err(io::Error::READ_EXACT_EOF) } else { Ok(()) }
    }

    /// Writes a number of bytes starting from a given offset.
    ///
    /// Returns the number of bytes written.
    ///
    /// The offset is relative to the start of the file and thus independent
    /// from the current cursor.
    ///
    /// The current file cursor is not affected by this function.
    ///
    /// When writing beyond the end of the file, the file is appropriately
    /// extended and the intermediate bytes are initialized with the value 0.
    ///
    /// Note that similar to [`File::write`], it is not an error to return a
    /// short write.
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize>;

    /// Writes a number of bytes starting from a given offset.
    ///
    /// Returns the number of bytes written.
    ///
    /// The offset is relative to the start of the file and thus independent
    /// from the current cursor.
    ///
    /// The current file cursor is not affected by this function.
    ///
    /// When writing beyond the end of the file, the file is appropriately
    /// extended and the intermediate bytes are initialized with the value 0.
    ///
    /// Note that similar to [`File::write_vectored`], it is not an error to return a
    /// short write.
    fn write_vectored_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize>;

    /// Attempts to write an entire buffer starting from a given offset.
    ///
    /// The offset is relative to the start of the file and thus independent
    /// from the current cursor.
    ///
    /// The current file cursor is not affected by this function.
    ///
    /// This method will continuously call [`write_at`] until there is no more data
    /// to be written or an error of non-[`io::ErrorKind::Interrupted`] kind is
    /// returned. This method will not return until the entire buffer has been
    /// successfully written or such an error occurs. The first error that is
    /// not of [`io::ErrorKind::Interrupted`] kind generated from this method will be
    /// returned.
    ///
    /// # Errors
    ///
    /// This function will return the first error of
    /// non-[`io::ErrorKind::Interrupted`] kind that [`write_at`] returns.
    ///
    /// [`write_at`]: FileExt::write_at
    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.write_at(buf, offset) {
                Ok(0) => {
                    return Err(io::Error::WRITE_ALL_EOF);
                }
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64
                }
                Err(ref e) if e.is_interrupted() => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Adjusts the flags associated with this file.
    ///
    /// This corresponds to the `fd_fdstat_set_flags` syscall.
    #[doc(alias = "fd_fdstat_set_flags")]
    #[cfg(target_env = "p1")]
    fn fdstat_set_flags(&self, flags: u16) -> io::Result<()>;

    /// Adjusts the rights associated with this file.
    ///
    /// This corresponds to the `fd_fdstat_set_rights` syscall.
    #[doc(alias = "fd_fdstat_set_rights")]
    #[cfg(target_env = "p1")]
    fn fdstat_set_rights(&self, rights: u64, inheriting: u64) -> io::Result<()>;

    /// Provides file advisory information on a file descriptor.
    ///
    /// This corresponds to the `fd_advise` syscall.
    #[doc(alias = "fd_advise")]
    #[cfg(target_env = "p1")]
    fn advise(&self, offset: u64, len: u64, advice: u8) -> io::Result<()>;

    /// Forces the allocation of space in a file.
    ///
    /// This corresponds to the `fd_allocate` syscall.
    #[doc(alias = "fd_allocate")]
    #[cfg(target_env = "p1")]
    fn allocate(&self, offset: u64, len: u64) -> io::Result<()>;

    /// Creates a directory.
    ///
    /// This corresponds to the `path_create_directory` syscall.
    #[doc(alias = "path_create_directory")]
    #[cfg(target_env = "p1")]
    fn create_directory<P: AsRef<Path>>(&self, dir: P) -> io::Result<()>;

    /// Unlinks a file.
    ///
    /// This corresponds to the `path_unlink_file` syscall.
    #[doc(alias = "path_unlink_file")]
    #[cfg(target_env = "p1")]
    fn remove_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()>;

    /// Removes a directory.
    ///
    /// This corresponds to the `path_remove_directory` syscall.
    #[doc(alias = "path_remove_directory")]
    #[cfg(target_env = "p1")]
    fn remove_directory<P: AsRef<Path>>(&self, path: P) -> io::Result<()>;
}

// FIXME: bind fd_fdstat_get - need to define a custom return type
// FIXME: bind fd_readdir - can't return `ReadDir` since we only have entry name
// FIXME: bind fd_filestat_set_times maybe? - on crates.io for unix
// FIXME: bind path_filestat_set_times maybe? - on crates.io for unix
// FIXME: bind poll_oneoff maybe? - probably should wait for I/O to settle
// FIXME: bind random_get maybe? - on crates.io for unix

impl FileExt for File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        self.as_inner().read_at(buf, offset)
    }

    fn read_buf_at(&self, buf: BorrowedCursor<'_>, offset: u64) -> io::Result<()> {
        self.as_inner().read_buf_at(buf, offset)
    }

    fn read_vectored_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        self.as_inner().read_vectored_at(bufs, offset)
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        self.as_inner().write_at(buf, offset)
    }

    fn write_vectored_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        self.as_inner().write_vectored_at(bufs, offset)
    }

    #[cfg(target_env = "p1")]
    fn fdstat_set_flags(&self, flags: u16) -> io::Result<()> {
        unsafe { wasi::fd_fdstat_set_flags(self.as_raw_fd() as wasi::Fd, flags).map_err(err2io) }
    }

    #[cfg(target_env = "p1")]
    fn fdstat_set_rights(&self, rights: u64, inheriting: u64) -> io::Result<()> {
        unsafe {
            wasi::fd_fdstat_set_rights(self.as_raw_fd() as wasi::Fd, rights, inheriting)
                .map_err(err2io)
        }
    }

    #[cfg(target_env = "p1")]
    fn advise(&self, offset: u64, len: u64, advice: u8) -> io::Result<()> {
        let advice = match advice {
            a if a == wasi::ADVICE_NORMAL.raw() => wasi::ADVICE_NORMAL,
            a if a == wasi::ADVICE_SEQUENTIAL.raw() => wasi::ADVICE_SEQUENTIAL,
            a if a == wasi::ADVICE_RANDOM.raw() => wasi::ADVICE_RANDOM,
            a if a == wasi::ADVICE_WILLNEED.raw() => wasi::ADVICE_WILLNEED,
            a if a == wasi::ADVICE_DONTNEED.raw() => wasi::ADVICE_DONTNEED,
            a if a == wasi::ADVICE_NOREUSE.raw() => wasi::ADVICE_NOREUSE,
            _ => {
                return Err(io::const_error!(
                    io::ErrorKind::InvalidInput,
                    "invalid parameter 'advice'",
                ));
            }
        };

        unsafe {
            wasi::fd_advise(self.as_raw_fd() as wasi::Fd, offset, len, advice).map_err(err2io)
        }
    }

    #[cfg(target_env = "p1")]
    fn allocate(&self, offset: u64, len: u64) -> io::Result<()> {
        unsafe { wasi::fd_allocate(self.as_raw_fd() as wasi::Fd, offset, len).map_err(err2io) }
    }

    #[cfg(target_env = "p1")]
    fn create_directory<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        let path = osstr2str(dir.as_ref().as_ref())?;
        unsafe { wasi::path_create_directory(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
    }

    #[cfg(target_env = "p1")]
    fn remove_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let path = osstr2str(path.as_ref().as_ref())?;
        unsafe { wasi::path_unlink_file(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
    }

    #[cfg(target_env = "p1")]
    fn remove_directory<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let path = osstr2str(path.as_ref().as_ref())?;
        unsafe { wasi::path_remove_directory(self.as_raw_fd() as wasi::Fd, path).map_err(err2io) }
    }
}

/// WASI-specific extensions to [`OpenOptions`].
pub trait OpenOptionsExt {
    /// Pass custom flags to the `flags` argument of `open`.
    fn custom_flags(&mut self, flags: i32) -> &mut Self;
}

impl OpenOptionsExt for OpenOptions {
    fn custom_flags(&mut self, flags: i32) -> &mut OpenOptions {
        self.as_inner_mut().custom_flags(flags);
        self
    }
}

/// WASI-specific extensions to [`fs::Metadata`].
pub trait MetadataExt {
    /// Returns the `st_dev` field of the internal `filestat_t`
    fn dev(&self) -> u64;
    /// Returns the `st_ino` field of the internal `filestat_t`
    fn ino(&self) -> u64;
    /// Returns the `st_nlink` field of the internal `filestat_t`
    fn nlink(&self) -> u64;
}

impl MetadataExt for fs::Metadata {
    fn dev(&self) -> u64 {
        self.as_inner().as_inner().st_dev
    }
    fn ino(&self) -> u64 {
        self.as_inner().as_inner().st_ino
    }
    fn nlink(&self) -> u64 {
        self.as_inner().as_inner().st_nlink
    }
}

/// WASI-specific extensions for [`fs::FileType`].
///
/// Adds support for special WASI file types such as block/character devices,
/// pipes, and sockets.
pub trait FileTypeExt {
    /// Returns `true` if this file type is a block device.
    fn is_block_device(&self) -> bool;
    /// Returns `true` if this file type is a character device.
    fn is_char_device(&self) -> bool;
    /// Returns `true` if this file type is any type of socket.
    fn is_socket(&self) -> bool;
}

impl FileTypeExt for fs::FileType {
    fn is_block_device(&self) -> bool {
        self.as_inner().is(libc::S_IFBLK)
    }
    fn is_char_device(&self) -> bool {
        self.as_inner().is(libc::S_IFCHR)
    }
    fn is_socket(&self) -> bool {
        self.as_inner().is(libc::S_IFSOCK)
    }
}

/// WASI-specific extension methods for [`fs::DirEntry`].
pub trait DirEntryExt {
    /// Returns the underlying `d_ino` field of the `dirent_t`
    fn ino(&self) -> u64;
}

impl DirEntryExt for fs::DirEntry {
    fn ino(&self) -> u64 {
        self.as_inner().ino()
    }
}

/// Creates a hard link.
///
/// This corresponds to the `path_link` syscall.
#[doc(alias = "path_link")]
#[cfg(target_env = "p1")]
pub fn link<P: AsRef<Path>, U: AsRef<Path>>(
    old_fd: &File,
    old_flags: u32,
    old_path: P,
    new_fd: &File,
    new_path: U,
) -> io::Result<()> {
    unsafe {
        wasi::path_link(
            old_fd.as_raw_fd() as wasi::Fd,
            old_flags,
            osstr2str(old_path.as_ref().as_ref())?,
            new_fd.as_raw_fd() as wasi::Fd,
            osstr2str(new_path.as_ref().as_ref())?,
        )
        .map_err(err2io)
    }
}

/// Renames a file or directory.
///
/// This corresponds to the `path_rename` syscall.
#[doc(alias = "path_rename")]
#[cfg(target_env = "p1")]
pub fn rename<P: AsRef<Path>, U: AsRef<Path>>(
    old_fd: &File,
    old_path: P,
    new_fd: &File,
    new_path: U,
) -> io::Result<()> {
    unsafe {
        wasi::path_rename(
            old_fd.as_raw_fd() as wasi::Fd,
            osstr2str(old_path.as_ref().as_ref())?,
            new_fd.as_raw_fd() as wasi::Fd,
            osstr2str(new_path.as_ref().as_ref())?,
        )
        .map_err(err2io)
    }
}

/// Creates a symbolic link.
///
/// This corresponds to the `path_symlink` syscall.
#[doc(alias = "path_symlink")]
#[cfg(target_env = "p1")]
pub fn symlink<P: AsRef<Path>, U: AsRef<Path>>(
    old_path: P,
    fd: &File,
    new_path: U,
) -> io::Result<()> {
    unsafe {
        wasi::path_symlink(
            osstr2str(old_path.as_ref().as_ref())?,
            fd.as_raw_fd() as wasi::Fd,
            osstr2str(new_path.as_ref().as_ref())?,
        )
        .map_err(err2io)
    }
}

/// Creates a symbolic link.
///
/// This is a convenience API similar to `std::os::unix::fs::symlink` and
/// `std::os::windows::fs::symlink_file` and `std::os::windows::fs::symlink_dir`.
pub fn symlink_path<P: AsRef<Path>, U: AsRef<Path>>(old_path: P, new_path: U) -> io::Result<()> {
    crate::sys::fs::symlink(old_path.as_ref(), new_path.as_ref())
}

#[cfg(target_env = "p1")]
fn osstr2str(f: &OsStr) -> io::Result<&str> {
    f.to_str().ok_or_else(|| io::const_error!(io::ErrorKind::Uncategorized, "input must be utf-8"))
}
