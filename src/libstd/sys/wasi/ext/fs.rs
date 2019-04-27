//! WASI-specific extensions to primitives in the `std::fs` module.

#![unstable(feature = "wasi_ext", issue = "0")]

use crate::fs::{self, File, Metadata, OpenOptions};
use crate::io::{self, IoSlice, IoSliceMut};
use crate::os::wasi::ffi::OsStrExt;
use crate::path::{Path, PathBuf};
use crate::sys_common::{AsInner, AsInnerMut, FromInner};

/// WASI-specific extensions to [`File`].
///
/// [`File`]: ../../../../std/fs/struct.File.html
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
    /// Note that similar to [`File::read_vectored`], it is not an error to
    /// return with a short read.
    ///
    /// [`File::read`]: ../../../../std/fs/struct.File.html#method.read_vectored
    fn read_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize>;

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
    ///
    /// [`File::write`]: ../../../../std/fs/struct.File.html#method.write_vectored
    fn write_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize>;

    /// Returns the current position within the file.
    ///
    /// This corresponds to the `__wasi_fd_tell` syscall and is similar to
    /// `seek` where you offset 0 bytes from the current position.
    fn tell(&self) -> io::Result<u64>;

    /// Adjust the flags associated with this file.
    ///
    /// This corresponds to the `__wasi_fd_fdstat_set_flags` syscall.
    fn fdstat_set_flags(&self, flags: u16) -> io::Result<()>;

    /// Adjust the rights associated with this file.
    ///
    /// This corresponds to the `__wasi_fd_fdstat_set_rights` syscall.
    fn fdstat_set_rights(&self, rights: u64, inheriting: u64) -> io::Result<()>;

    /// Provide file advisory information on a file descriptor.
    ///
    /// This corresponds to the `__wasi_fd_advise` syscall.
    fn advise(&self, offset: u64, len: u64, advice: u8) -> io::Result<()>;

    /// Force the allocation of space in a file.
    ///
    /// This corresponds to the `__wasi_fd_allocate` syscall.
    fn allocate(&self, offset: u64, len: u64) -> io::Result<()>;

    /// Create a directory.
    ///
    /// This corresponds to the `__wasi_path_create_directory` syscall.
    fn create_directory<P: AsRef<Path>>(&self, dir: P) -> io::Result<()>;

    /// Read the contents of a symbolic link.
    ///
    /// This corresponds to the `__wasi_path_readlink` syscall.
    fn read_link<P: AsRef<Path>>(&self, path: P) -> io::Result<PathBuf>;

    /// Return the attributes of a file or directory.
    ///
    /// This corresponds to the `__wasi_path_filestat_get` syscall.
    fn metadata_at<P: AsRef<Path>>(&self, lookup_flags: u32, path: P) -> io::Result<Metadata>;

    /// Unlink a file.
    ///
    /// This corresponds to the `__wasi_path_unlink_file` syscall.
    fn remove_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()>;

    /// Remove a directory.
    ///
    /// This corresponds to the `__wasi_path_remove_directory` syscall.
    fn remove_directory<P: AsRef<Path>>(&self, path: P) -> io::Result<()>;
}

// FIXME: bind __wasi_fd_fdstat_get - need to define a custom return type
// FIXME: bind __wasi_fd_readdir - can't return `ReadDir` since we only have entry name
// FIXME: bind __wasi_fd_filestat_set_times maybe? - on crates.io for unix
// FIXME: bind __wasi_path_filestat_set_times maybe? - on crates.io for unix
// FIXME: bind __wasi_poll_oneoff maybe? - probably should wait for I/O to settle
// FIXME: bind __wasi_random_get maybe? - on crates.io for unix

impl FileExt for fs::File {
    fn read_at(&self, bufs: &mut [IoSliceMut<'_>], offset: u64) -> io::Result<usize> {
        self.as_inner().fd().pread(bufs, offset)
    }

    fn write_at(&self, bufs: &[IoSlice<'_>], offset: u64) -> io::Result<usize> {
        self.as_inner().fd().pwrite(bufs, offset)
    }

    fn tell(&self) -> io::Result<u64> {
        self.as_inner().fd().tell()
    }

    fn fdstat_set_flags(&self, flags: u16) -> io::Result<()> {
        self.as_inner().fd().set_flags(flags)
    }

    fn fdstat_set_rights(&self, rights: u64, inheriting: u64) -> io::Result<()> {
        self.as_inner().fd().set_rights(rights, inheriting)
    }

    fn advise(&self, offset: u64, len: u64, advice: u8) -> io::Result<()> {
        self.as_inner().fd().advise(offset, len, advice)
    }

    fn allocate(&self, offset: u64, len: u64) -> io::Result<()> {
        self.as_inner().fd().allocate(offset, len)
    }

    fn create_directory<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        self.as_inner()
            .fd()
            .create_directory(dir.as_ref().as_os_str().as_bytes())
    }

    fn read_link<P: AsRef<Path>>(&self, path: P) -> io::Result<PathBuf> {
        self.as_inner().read_link(path.as_ref())
    }

    fn metadata_at<P: AsRef<Path>>(&self, lookup_flags: u32, path: P) -> io::Result<Metadata> {
        let m = self.as_inner().metadata_at(lookup_flags, path.as_ref())?;
        Ok(FromInner::from_inner(m))
    }

    fn remove_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        self.as_inner()
            .fd()
            .unlink_file(path.as_ref().as_os_str().as_bytes())
    }

    fn remove_directory<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        self.as_inner()
            .fd()
            .remove_directory(path.as_ref().as_os_str().as_bytes())
    }
}

/// WASI-specific extensions to [`fs::OpenOptions`].
///
/// [`fs::OpenOptions`]: ../../../../std/fs/struct.OpenOptions.html
pub trait OpenOptionsExt {
    /// Pass custom `dirflags` argument to `__wasi_path_open`.
    ///
    /// This option configures the `dirflags` argument to the
    /// `__wasi_path_open` syscall which `OpenOptions` will eventually call. The
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
    /// field of `__wasi_path_open`.
    ///
    /// This option is by default `false`
    fn dsync(&mut self, dsync: bool) -> &mut Self;

    /// Indicates whether `__WASI_FDFLAG_NONBLOCK` is passed in the `fs_flags`
    /// field of `__wasi_path_open`.
    ///
    /// This option is by default `false`
    fn nonblock(&mut self, nonblock: bool) -> &mut Self;

    /// Indicates whether `__WASI_FDFLAG_RSYNC` is passed in the `fs_flags`
    /// field of `__wasi_path_open`.
    ///
    /// This option is by default `false`
    fn rsync(&mut self, rsync: bool) -> &mut Self;

    /// Indicates whether `__WASI_FDFLAG_SYNC` is passed in the `fs_flags`
    /// field of `__wasi_path_open`.
    ///
    /// This option is by default `false`
    fn sync(&mut self, sync: bool) -> &mut Self;

    /// Indicates the value that should be passed in for the `fs_rights_base`
    /// parameter of `__wasi_path_open`.
    ///
    /// This option defaults based on the `read` and `write` configuration of
    /// this `OpenOptions` builder. If this method is called, however, the
    /// exact mask passed in will be used instead.
    fn fs_rights_base(&mut self, rights: u64) -> &mut Self;

    /// Indicates the value that should be passed in for the
    /// `fs_rights_inheriting` parameter of `__wasi_path_open`.
    ///
    /// The default for this option is the same value as what will be passed
    /// for the `fs_rights_base` parameter but if this method is called then
    /// the specified value will be used instead.
    fn fs_rights_inheriting(&mut self, rights: u64) -> &mut Self;

    /// Open a file or directory.
    ///
    /// This corresponds to the `__wasi_path_open` syscall.
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
///
/// [`fs::Metadata`]: ../../../../std/fs/struct.Metadata.html
pub trait MetadataExt {
    /// Returns the `st_dev` field of the internal `__wasi_filestat_t`
    fn dev(&self) -> u64;
    /// Returns the `st_ino` field of the internal `__wasi_filestat_t`
    fn ino(&self) -> u64;
    /// Returns the `st_nlink` field of the internal `__wasi_filestat_t`
    fn nlink(&self) -> u32;
    /// Returns the `st_atim` field of the internal `__wasi_filestat_t`
    fn atim(&self) -> u64;
    /// Returns the `st_mtim` field of the internal `__wasi_filestat_t`
    fn mtim(&self) -> u64;
    /// Returns the `st_ctim` field of the internal `__wasi_filestat_t`
    fn ctim(&self) -> u64;
}

impl MetadataExt for fs::Metadata {
    fn dev(&self) -> u64 {
        self.as_inner().as_wasi().st_dev
    }
    fn ino(&self) -> u64 {
        self.as_inner().as_wasi().st_ino
    }
    fn nlink(&self) -> u32 {
        self.as_inner().as_wasi().st_nlink
    }
    fn atim(&self) -> u64 {
        self.as_inner().as_wasi().st_atim
    }
    fn mtim(&self) -> u64 {
        self.as_inner().as_wasi().st_mtim
    }
    fn ctim(&self) -> u64 {
        self.as_inner().as_wasi().st_ctim
    }
}

/// WASI-specific extensions for [`FileType`].
///
/// Adds support for special WASI file types such as block/character devices,
/// pipes, and sockets.
///
/// [`FileType`]: ../../../../std/fs/struct.FileType.html
pub trait FileTypeExt {
    /// Returns `true` if this file type is a block device.
    fn is_block_device(&self) -> bool;
    /// Returns `true` if this file type is a character device.
    fn is_character_device(&self) -> bool;
    /// Returns `true` if this file type is a socket datagram.
    fn is_socket_dgram(&self) -> bool;
    /// Returns `true` if this file type is a socket stream.
    fn is_socket_stream(&self) -> bool;
}

impl FileTypeExt for fs::FileType {
    fn is_block_device(&self) -> bool {
        self.as_inner().bits() == libc::__WASI_FILETYPE_BLOCK_DEVICE
    }
    fn is_character_device(&self) -> bool {
        self.as_inner().bits() == libc::__WASI_FILETYPE_CHARACTER_DEVICE
    }
    fn is_socket_dgram(&self) -> bool {
        self.as_inner().bits() == libc::__WASI_FILETYPE_SOCKET_DGRAM
    }
    fn is_socket_stream(&self) -> bool {
        self.as_inner().bits() == libc::__WASI_FILETYPE_SOCKET_STREAM
    }
}

/// WASI-specific extension methods for [`fs::DirEntry`].
///
/// [`fs::DirEntry`]: ../../../../std/fs/struct.DirEntry.html
pub trait DirEntryExt {
    /// Returns the underlying `d_ino` field of the `__wasi_dirent_t`
    fn ino(&self) -> u64;
}

impl DirEntryExt for fs::DirEntry {
    fn ino(&self) -> u64 {
        self.as_inner().ino()
    }
}

/// Create a hard link.
///
/// This corresponds to the `__wasi_path_link` syscall.
pub fn link<P: AsRef<Path>, U: AsRef<Path>>(
    old_fd: &File,
    old_flags: u32,
    old_path: P,
    new_fd: &File,
    new_path: U,
) -> io::Result<()> {
    old_fd.as_inner().fd().link(
        old_flags,
        old_path.as_ref().as_os_str().as_bytes(),
        new_fd.as_inner().fd(),
        new_path.as_ref().as_os_str().as_bytes(),
    )
}

/// Rename a file or directory.
///
/// This corresponds to the `__wasi_path_rename` syscall.
pub fn rename<P: AsRef<Path>, U: AsRef<Path>>(
    old_fd: &File,
    old_path: P,
    new_fd: &File,
    new_path: U,
) -> io::Result<()> {
    old_fd.as_inner().fd().rename(
        old_path.as_ref().as_os_str().as_bytes(),
        new_fd.as_inner().fd(),
        new_path.as_ref().as_os_str().as_bytes(),
    )
}

/// Create a symbolic link.
///
/// This corresponds to the `__wasi_path_symlink` syscall.
pub fn symlink<P: AsRef<Path>, U: AsRef<Path>>(
    old_path: P,
    fd: &File,
    new_path: U,
) -> io::Result<()> {
    fd.as_inner().fd().symlink(
        old_path.as_ref().as_os_str().as_bytes(),
        new_path.as_ref().as_os_str().as_bytes(),
    )
}
