//! WASI-specific extensions to primitives in the [`std::fs`] module.
//!
//! [`std::fs`]: crate::fs

#![unstable(feature = "wasi_ext", issue = "71213")]

// Used for `File::read` on intra-doc links
#[allow(unused_imports)]
use io::{Read, Write};

use crate::ffi::OsStr;
use crate::fs::{self, File, Metadata, OpenOptions};
use crate::io::{self, IoSlice, IoSliceMut};
use crate::path::{Path, PathBuf};
use crate::sys_common::{AsInner, AsInnerMut, FromInner};

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
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let bufs = &mut [IoSliceMut::new(buf)];
        self.read_vectored_at(bufs, offset)
    }

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
    #[stable(feature = "rw_exact_all_at", since = "1.33.0")]
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
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        let bufs = &[IoSlice::new(buf)];
        self.write_vectored_at(bufs, offset)
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
    #[stable(feature = "rw_exact_all_at", since = "1.33.0")]
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

    /// Returns the current position within the file.
    ///
    /// This corresponds to the `fd_tell` syscall and is similar to
    /// `seek` where you offset 0 bytes from the current position.
    #[doc(alias = "fd_tell")]
    fn tell(&self) -> io::Result<u64>;

    /// Adjusts the flags associated with this file.
    ///
    /// This corresponds to the `fd_fdstat_set_flags` syscall.
    #[doc(alias = "fd_fdstat_set_flags")]
    fn fdstat_set_flags(&self, flags: u16) -> io::Result<()>;

    /// Adjusts the rights associated with this file.
    ///
    /// This corresponds to the `fd_fdstat_set_rights` syscall.
    #[doc(alias = "fd_fdstat_set_rights")]
    fn fdstat_set_rights(&self, rights: u64, inheriting: u64) -> io::Result<()>;

    /// Provides file advisory information on a file descriptor.
    ///
    /// This corresponds to the `fd_advise` syscall.
    #[doc(alias = "fd_advise")]
    fn advise(&self, offset: u64, len: u64, advice: u8) -> io::Result<()>;

    /// Forces the allocation of space in a file.
    ///
    /// This corresponds to the `fd_allocate` syscall.
    #[doc(alias = "fd_allocate")]
    fn allocate(&self, offset: u64, len: u64) -> io::Result<()>;

    /// Creates a directory.
    ///
    /// This corresponds to the `path_create_directory` syscall.
    #[doc(alias = "path_create_directory")]
    fn create_directory<P: AsRef<Path>>(&self, dir: P) -> io::Result<()>;

    /// Reads the contents of a symbolic link.
    ///
    /// This corresponds to the `path_readlink` syscall.
    #[doc(alias = "path_readlink")]
    fn read_link<P: AsRef<Path>>(&self, path: P) -> io::Result<PathBuf>;

    /// Returns the attributes of a file or directory.
    ///
    /// This corresponds to the `path_filestat_get` syscall.
    #[doc(alias = "path_filestat_get")]
    fn metadata_at<P: AsRef<Path>>(&self, lookup_flags: u32, path: P) -> io::Result<Metadata>;

    /// Unlinks a file.
    ///
    /// This corresponds to the `path_unlink_file` syscall.
    #[doc(alias = "path_unlink_file")]
    fn remove_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()>;

    /// Removes a directory.
    ///
    /// This corresponds to the `path_remove_directory` syscall.
    #[doc(alias = "path_remove_directory")]
    fn remove_directory<P: AsRef<Path>>(&self, path: P) -> io::Result<()>;
}

// FIXME: bind fd_fdstat_get - need to define a custom return type
// FIXME: bind fd_readdir - can't return `ReadDir` since we only have entry name
// FIXME: bind fd_filestat_set_times maybe? - on crates.io for unix
// FIXME: bind path_filestat_set_times maybe? - on crates.io for unix
// FIXME: bind poll_oneoff maybe? - probably should wait for I/O to settle
// FIXME: bind random_get maybe? - on crates.io for unix

impl FileExt for fs::File {
    fn read_vectored_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        self.as_inner().as_inner().pread(bufs, offset)
    }

    fn write_vectored_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        self.as_inner().as_inner().pwrite(bufs, offset)
    }

    fn tell(&self) -> io::Result<u64> {
        self.as_inner().as_inner().tell()
    }

    fn fdstat_set_flags(&self, flags: u16) -> io::Result<()> {
        self.as_inner().as_inner().set_flags(flags)
    }

    fn fdstat_set_rights(&self, rights: u64, inheriting: u64) -> io::Result<()> {
        self.as_inner().as_inner().set_rights(rights, inheriting)
    }

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

        self.as_inner().as_inner().advise(offset, len, advice)
    }

    fn allocate(&self, offset: u64, len: u64) -> io::Result<()> {
        self.as_inner().as_inner().allocate(offset, len)
    }

    fn create_directory<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        self.as_inner().as_inner().create_directory(osstr2str(dir.as_ref().as_ref())?)
    }

    fn read_link<P: AsRef<Path>>(&self, path: P) -> io::Result<PathBuf> {
        self.as_inner().read_link(path.as_ref())
    }

    fn metadata_at<P: AsRef<Path>>(&self, lookup_flags: u32, path: P) -> io::Result<Metadata> {
        let m = self.as_inner().metadata_at(lookup_flags, path.as_ref())?;
        Ok(FromInner::from_inner(m))
    }

    fn remove_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        self.as_inner().as_inner().unlink_file(osstr2str(path.as_ref().as_ref())?)
    }

    fn remove_directory<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        self.as_inner().as_inner().remove_directory(osstr2str(path.as_ref().as_ref())?)
    }
}

/// WASI-specific extensions to [`fs::OpenOptions`].
pub trait OpenOptionsExt {
    /// Pass custom `dirflags` argument to `path_open`.
    ///
    /// This option configures the `dirflags` argument to the
    /// `path_open` syscall which `OpenOptions` will eventually call. The
    /// `dirflags` argument configures how the file is looked up, currently
    /// primarily affecting whether symlinks are followed or not.
    ///
    /// By default this value is `__WASI_LOOKUP_SYMLINK_FOLLOW`, or symlinks are
    /// followed. You can call this method with 0 to disable following symlinks
    fn lookup_flags(&mut self, flags: u32) -> &mut Self;

    /// Indicates whether `OpenOptions` must open a directory or not.
    ///
    /// This method will configure whether the `__WASI_O_DIRECTORY` flag is
    /// passed when opening a file. When passed it will require that the opened
    /// path is a directory.
    ///
    /// This option is by default `false`
    fn directory(&mut self, dir: bool) -> &mut Self;

    /// Indicates whether `__WASI_FDFLAG_DSYNC` is passed in the `fs_flags`
    /// field of `path_open`.
    ///
    /// This option is by default `false`
    fn dsync(&mut self, dsync: bool) -> &mut Self;

    /// Indicates whether `__WASI_FDFLAG_NONBLOCK` is passed in the `fs_flags`
    /// field of `path_open`.
    ///
    /// This option is by default `false`
    fn nonblock(&mut self, nonblock: bool) -> &mut Self;

    /// Indicates whether `__WASI_FDFLAG_RSYNC` is passed in the `fs_flags`
    /// field of `path_open`.
    ///
    /// This option is by default `false`
    fn rsync(&mut self, rsync: bool) -> &mut Self;

    /// Indicates whether `__WASI_FDFLAG_SYNC` is passed in the `fs_flags`
    /// field of `path_open`.
    ///
    /// This option is by default `false`
    fn sync(&mut self, sync: bool) -> &mut Self;

    /// Indicates the value that should be passed in for the `fs_rights_base`
    /// parameter of `path_open`.
    ///
    /// This option defaults based on the `read` and `write` configuration of
    /// this `OpenOptions` builder. If this method is called, however, the
    /// exact mask passed in will be used instead.
    fn fs_rights_base(&mut self, rights: u64) -> &mut Self;

    /// Indicates the value that should be passed in for the
    /// `fs_rights_inheriting` parameter of `path_open`.
    ///
    /// The default for this option is the same value as what will be passed
    /// for the `fs_rights_base` parameter but if this method is called then
    /// the specified value will be used instead.
    fn fs_rights_inheriting(&mut self, rights: u64) -> &mut Self;

    /// Open a file or directory.
    ///
    /// This corresponds to the `path_open` syscall.
    #[doc(alias = "path_open")]
    fn open_at<P: AsRef<Path>>(&self, file: &File, path: P) -> io::Result<File>;
}

impl OpenOptionsExt for OpenOptions {
    fn lookup_flags(&mut self, flags: u32) -> &mut OpenOptions {
        self.as_inner_mut().lookup_flags(flags);
        self
    }

    fn directory(&mut self, dir: bool) -> &mut OpenOptions {
        self.as_inner_mut().directory(dir);
        self
    }

    fn dsync(&mut self, enabled: bool) -> &mut OpenOptions {
        self.as_inner_mut().dsync(enabled);
        self
    }

    fn nonblock(&mut self, enabled: bool) -> &mut OpenOptions {
        self.as_inner_mut().nonblock(enabled);
        self
    }

    fn rsync(&mut self, enabled: bool) -> &mut OpenOptions {
        self.as_inner_mut().rsync(enabled);
        self
    }

    fn sync(&mut self, enabled: bool) -> &mut OpenOptions {
        self.as_inner_mut().sync(enabled);
        self
    }

    fn fs_rights_base(&mut self, rights: u64) -> &mut OpenOptions {
        self.as_inner_mut().fs_rights_base(rights);
        self
    }

    fn fs_rights_inheriting(&mut self, rights: u64) -> &mut OpenOptions {
        self.as_inner_mut().fs_rights_inheriting(rights);
        self
    }

    fn open_at<P: AsRef<Path>>(&self, file: &File, path: P) -> io::Result<File> {
        let inner = file.as_inner().open_at(path.as_ref(), self.as_inner())?;
        Ok(File::from_inner(inner))
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
    /// Returns the `st_size` field of the internal `filestat_t`
    fn size(&self) -> u64;
    /// Returns the `st_atim` field of the internal `filestat_t`
    fn atim(&self) -> u64;
    /// Returns the `st_mtim` field of the internal `filestat_t`
    fn mtim(&self) -> u64;
    /// Returns the `st_ctim` field of the internal `filestat_t`
    fn ctim(&self) -> u64;
}

impl MetadataExt for fs::Metadata {
    fn dev(&self) -> u64 {
        self.as_inner().as_wasi().dev
    }
    fn ino(&self) -> u64 {
        self.as_inner().as_wasi().ino
    }
    fn nlink(&self) -> u64 {
        self.as_inner().as_wasi().nlink
    }
    fn size(&self) -> u64 {
        self.as_inner().as_wasi().size
    }
    fn atim(&self) -> u64 {
        self.as_inner().as_wasi().atim
    }
    fn mtim(&self) -> u64 {
        self.as_inner().as_wasi().mtim
    }
    fn ctim(&self) -> u64 {
        self.as_inner().as_wasi().ctim
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
    /// Returns `true` if this file type is a socket datagram.
    fn is_socket_dgram(&self) -> bool;
    /// Returns `true` if this file type is a socket stream.
    fn is_socket_stream(&self) -> bool;
    /// Returns `true` if this file type is any type of socket.
    fn is_socket(&self) -> bool {
        self.is_socket_stream() || self.is_socket_dgram()
    }
}

impl FileTypeExt for fs::FileType {
    fn is_block_device(&self) -> bool {
        self.as_inner().bits() == wasi::FILETYPE_BLOCK_DEVICE
    }
    fn is_char_device(&self) -> bool {
        self.as_inner().bits() == wasi::FILETYPE_CHARACTER_DEVICE
    }
    fn is_socket_dgram(&self) -> bool {
        self.as_inner().bits() == wasi::FILETYPE_SOCKET_DGRAM
    }
    fn is_socket_stream(&self) -> bool {
        self.as_inner().bits() == wasi::FILETYPE_SOCKET_STREAM
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
pub fn link<P: AsRef<Path>, U: AsRef<Path>>(
    old_fd: &File,
    old_flags: u32,
    old_path: P,
    new_fd: &File,
    new_path: U,
) -> io::Result<()> {
    old_fd.as_inner().as_inner().link(
        old_flags,
        osstr2str(old_path.as_ref().as_ref())?,
        new_fd.as_inner().as_inner(),
        osstr2str(new_path.as_ref().as_ref())?,
    )
}

/// Renames a file or directory.
///
/// This corresponds to the `path_rename` syscall.
#[doc(alias = "path_rename")]
pub fn rename<P: AsRef<Path>, U: AsRef<Path>>(
    old_fd: &File,
    old_path: P,
    new_fd: &File,
    new_path: U,
) -> io::Result<()> {
    old_fd.as_inner().as_inner().rename(
        osstr2str(old_path.as_ref().as_ref())?,
        new_fd.as_inner().as_inner(),
        osstr2str(new_path.as_ref().as_ref())?,
    )
}

/// Creates a symbolic link.
///
/// This corresponds to the `path_symlink` syscall.
#[doc(alias = "path_symlink")]
pub fn symlink<P: AsRef<Path>, U: AsRef<Path>>(
    old_path: P,
    fd: &File,
    new_path: U,
) -> io::Result<()> {
    fd.as_inner()
        .as_inner()
        .symlink(osstr2str(old_path.as_ref().as_ref())?, osstr2str(new_path.as_ref().as_ref())?)
}

/// Creates a symbolic link.
///
/// This is a convenience API similar to `std::os::unix::fs::symlink` and
/// `std::os::windows::fs::symlink_file` and `std::os::windows::fs::symlink_dir`.
pub fn symlink_path<P: AsRef<Path>, U: AsRef<Path>>(old_path: P, new_path: U) -> io::Result<()> {
    crate::sys::fs::symlink(old_path.as_ref(), new_path.as_ref())
}

fn osstr2str(f: &OsStr) -> io::Result<&str> {
    f.to_str().ok_or_else(|| io::const_error!(io::ErrorKind::Uncategorized, "input must be utf-8"))
}
