use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fs::TryLockError;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::mem::ManuallyDrop;
use crate::os::raw::c_int;
use crate::os::wasi::ffi::{OsStrExt, OsStringExt};
use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::path::{Path, PathBuf};
use crate::sync::Arc;
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::fd::WasiFd;
pub use crate::sys::fs::common::exists;
use crate::sys::time::SystemTime;
use crate::sys::{unsupported, unsupported_err};
use crate::sys_common::{AsInner, FromInner, IntoInner, ignore_notfound};
use crate::{fmt, iter, ptr};

pub struct File {
    fd: WasiFd,
}

#[derive(Clone)]
pub struct FileAttr {
    stat: libc::stat,
}

pub struct ReadDir {
    inner: Arc<ReadDirInner>,
    done: bool,
}

struct ReadDirInner {
    root: PathBuf,
    dirp: c::Dirp,
}

pub struct DirEntry {
    inner: Arc<ReadDirInner>,
    #[cfg(target_env = "p1")]
    d_ino: libc::ino_t,
    d_type: libc::c_uchar,
    name: CString,
}

#[derive(Clone, Debug, Default)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    custom_flags: libc::c_int,

    #[cfg(target_env = "p1")]
    use_wasip1: bool,
    #[cfg(target_env = "p1")]
    wasip1_dirflags: Option<wasi::Lookupflags>,
    #[cfg(target_env = "p1")]
    wasip1_fdflags: wasi::Fdflags,
    #[cfg(target_env = "p1")]
    wasip1_rights_base: Option<wasi::Rights>,
    #[cfg(target_env = "p1")]
    wasip1_rights_inheriting: Option<wasi::Rights>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    readonly: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    accessed: Option<SystemTime>,
    modified: Option<SystemTime>,
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FileType {
    mode: libc::mode_t,
}

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.stat.st_size as u64
    }

    pub fn perm(&self) -> FilePermissions {
        // not currently implemented in wasi yet
        FilePermissions { readonly: false }
    }

    pub fn file_type(&self) -> FileType {
        FileType { mode: self.stat.st_mode }
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_timespec(self.stat.st_mtim))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_timespec(self.stat.st_atim))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_timespec(self.stat.st_ctim))
    }

    #[cfg(target_env = "p1")]
    pub(crate) fn as_libc(&self) -> &libc::stat {
        &self.stat
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.readonly
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        self.readonly = readonly;
    }
}

impl FileTimes {
    pub fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = Some(t);
    }

    pub fn set_modified(&mut self, t: SystemTime) {
        self.modified = Some(t);
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.mode == libc::S_IFDIR
    }

    pub fn is_file(&self) -> bool {
        self.mode == libc::S_IFREG
    }

    pub fn is_symlink(&self) -> bool {
        self.mode == libc::S_IFLNK
    }

    #[cfg(target_env = "p1")]
    pub(crate) fn mode(&self) -> libc::mode_t {
        self.mode
    }
}

impl ReadDir {
    fn new(dir: File, root: PathBuf) -> io::Result<ReadDir> {
        let dirp = c::Dirp::new(dir)?;
        Ok(ReadDir { inner: Arc::new(ReadDirInner { dirp, root }), done: false })
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadDir").finish_non_exhaustive()
    }
}

impl core::iter::FusedIterator for ReadDir {}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        if self.done {
            return None;
        }

        loop {
            let entry_ptr = match self.inner.dirp.readdir() {
                Ok(Some(ptr)) => ptr,
                Ok(None) => {
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };

            unsafe {
                let name = CStr::from_ptr((&raw const (*entry_ptr).d_name).cast());
                let name_bytes = name.to_bytes();
                if name_bytes == b"." || name_bytes == b".." {
                    continue;
                }

                return Some(Ok(DirEntry {
                    d_type: (*entry_ptr).d_type,
                    #[cfg(target_env = "p1")]
                    d_ino: (*entry_ptr).d_ino,
                    name: name.to_owned(),
                    inner: Arc::clone(&self.inner),
                }));
            }
        }
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.inner.root.join(self.file_name_os_str())
    }

    pub fn file_name(&self) -> OsString {
        self.file_name_os_str().to_owned()
    }

    fn file_name_os_str(&self) -> &OsStr {
        OsStr::from_bytes(self.name.to_bytes())
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        c::fstatat(self.dir_fd(), &self.name, libc::AT_SYMLINK_NOFOLLOW)
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        match self.d_type {
            libc::DT_CHR => Ok(FileType { mode: libc::S_IFCHR }),
            libc::DT_LNK => Ok(FileType { mode: libc::S_IFLNK }),
            libc::DT_REG => Ok(FileType { mode: libc::S_IFREG }),
            libc::DT_DIR => Ok(FileType { mode: libc::S_IFDIR }),
            libc::DT_BLK => Ok(FileType { mode: libc::S_IFBLK }),
            _ => self.metadata().map(|m| m.file_type()),
        }
    }

    #[cfg(target_env = "p1")]
    pub fn ino(&self) -> wasi::Inode {
        self.d_ino
    }

    fn dir_fd(&self) -> BorrowedFd<'_> {
        self.inner.dirp.as_fd()
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions::default()
    }

    pub fn read(&mut self, read: bool) {
        self.read = read;
    }

    pub fn write(&mut self, write: bool) {
        self.write = write;
    }

    pub fn truncate(&mut self, truncate: bool) {
        self.truncate = truncate;
    }

    pub fn create(&mut self, create: bool) {
        self.create = create;
    }

    pub fn create_new(&mut self, create_new: bool) {
        self.create_new = create_new;
    }

    pub fn append(&mut self, append: bool) {
        self.append = append;
        #[cfg(target_env = "p1")]
        self.wasip1_fdflag(wasi::FDFLAGS_APPEND, append);
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_dsync(&mut self, set: bool) {
        self.wasip1_fdflag(wasi::FDFLAGS_DSYNC, set);
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_nonblock(&mut self, set: bool) {
        self.wasip1_fdflag(wasi::FDFLAGS_NONBLOCK, set);
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_rsync(&mut self, set: bool) {
        self.wasip1_fdflag(wasi::FDFLAGS_RSYNC, set);
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_sync(&mut self, set: bool) {
        self.wasip1_fdflag(wasi::FDFLAGS_SYNC, set);
    }

    #[cfg(target_env = "p1")]
    fn wasip1_fdflag(&mut self, bit: wasi::Fdflags, set: bool) {
        self.use_wasip1 = true;
        if set {
            self.wasip1_fdflags |= bit;
        } else {
            self.wasip1_fdflags &= !bit;
        }
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_fs_rights_base(&mut self, rights: wasi::Rights) {
        self.use_wasip1 = true;
        self.wasip1_rights_base = Some(rights);
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_fs_rights_inheriting(&mut self, rights: wasi::Rights) {
        self.use_wasip1 = true;
        self.wasip1_rights_inheriting = Some(rights);
    }

    #[cfg(target_env = "p1")]
    fn wasip1_rights_base(&self) -> wasi::Rights {
        if let Some(rights) = self.wasip1_rights_base {
            return rights;
        }

        // If rights haven't otherwise been specified try to pick a reasonable
        // set. This can always be overridden by users via extension traits, and
        // implementations may give us fewer rights silently than we ask for. So
        // given that, just look at `read` and `write` and bucket permissions
        // based on that.
        let mut base = 0;
        if self.read {
            base |= wasi::RIGHTS_FD_READ;
            base |= wasi::RIGHTS_FD_READDIR;
        }
        if self.write || self.append {
            base |= wasi::RIGHTS_FD_WRITE;
            base |= wasi::RIGHTS_FD_DATASYNC;
            base |= wasi::RIGHTS_FD_ALLOCATE;
            base |= wasi::RIGHTS_FD_FILESTAT_SET_SIZE;
        }

        // FIXME: some of these should probably be read-only or write-only...
        base |= wasi::RIGHTS_FD_ADVISE;
        base |= wasi::RIGHTS_FD_FDSTAT_SET_FLAGS;
        base |= wasi::RIGHTS_FD_FILESTAT_GET;
        base |= wasi::RIGHTS_FD_FILESTAT_SET_TIMES;
        base |= wasi::RIGHTS_FD_SEEK;
        base |= wasi::RIGHTS_FD_SYNC;
        base |= wasi::RIGHTS_FD_TELL;
        base |= wasi::RIGHTS_PATH_CREATE_DIRECTORY;
        base |= wasi::RIGHTS_PATH_CREATE_FILE;
        base |= wasi::RIGHTS_PATH_FILESTAT_GET;
        base |= wasi::RIGHTS_PATH_LINK_SOURCE;
        base |= wasi::RIGHTS_PATH_LINK_TARGET;
        base |= wasi::RIGHTS_PATH_OPEN;
        base |= wasi::RIGHTS_PATH_READLINK;
        base |= wasi::RIGHTS_PATH_REMOVE_DIRECTORY;
        base |= wasi::RIGHTS_PATH_RENAME_SOURCE;
        base |= wasi::RIGHTS_PATH_RENAME_TARGET;
        base |= wasi::RIGHTS_PATH_SYMLINK;
        base |= wasi::RIGHTS_PATH_UNLINK_FILE;
        base |= wasi::RIGHTS_POLL_FD_READWRITE;

        base
    }

    #[cfg(target_env = "p1")]
    fn wasip1_rights_inheriting(&self) -> wasi::Rights {
        self.wasip1_rights_inheriting.unwrap_or_else(|| self.wasip1_rights_base())
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_lookup_flags(&mut self, flags: wasi::Lookupflags) {
        self.use_wasip1 = true;
        self.wasip1_dirflags = Some(flags);
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_directory(&mut self, enable: bool) {
        if enable {
            self.custom_flags |= libc::O_DIRECTORY;
        } else {
            self.custom_flags &= !libc::O_DIRECTORY;
        }
        self.use_wasip1 = true;
    }

    pub fn custom_flags(&mut self, flags: libc::c_int) {
        self.custom_flags = flags;
    }

    fn open_flags(&self) -> io::Result<libc::c_int> {
        Ok(self.get_access_mode()? | self.get_creation_mode()? | self.custom_flags)
    }

    fn get_access_mode(&self) -> io::Result<c_int> {
        match (self.read, self.write, self.append) {
            (true, false, false) => Ok(libc::O_RDONLY),
            (false, true, false) => Ok(libc::O_WRONLY),
            (true, true, false) => Ok(libc::O_RDWR),
            (false, _, true) => Ok(libc::O_WRONLY | libc::O_APPEND),
            (true, _, true) => Ok(libc::O_RDWR | libc::O_APPEND),
            (false, false, false) => {
                // If no access mode is set, check if any creation flags are set
                // to provide a more descriptive error message
                if self.create || self.create_new || self.truncate {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "creating or truncating a file requires write or append access",
                    ))
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "must specify at least one of read, write, or append access",
                    ))
                }
            }
        }
    }

    fn get_creation_mode(&self) -> io::Result<c_int> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "creating or truncating a file requires write or append access",
                    ));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "creating or truncating a file requires write or append access",
                    ));
                }
            }
        }

        Ok(match (self.create, self.truncate, self.create_new) {
            (false, false, false) => 0,
            (true, false, false) => libc::O_CREAT,
            (false, true, false) => libc::O_TRUNC,
            (true, true, false) => libc::O_CREAT | libc::O_TRUNC,
            (_, _, true) => libc::O_CREAT | libc::O_EXCL,
        })
    }

    fn open_mode(&self) -> libc::c_int {
        0o666
    }

    #[cfg(target_env = "p1")]
    pub fn wasip1_oflags(&self) -> wasi::Oflags {
        let mut flags = 0;
        if self.create {
            flags |= wasi::OFLAGS_CREAT;
        }
        if self.create_new {
            flags |= wasi::OFLAGS_CREAT | wasi::OFLAGS_EXCL;
        }
        if self.truncate {
            flags |= wasi::OFLAGS_TRUNC;
        }
        if self.custom_flags == libc::O_DIRECTORY {
            flags |= wasi::OFLAGS_DIRECTORY;
        }
        flags
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let (dir, file) = open_parent(path)?;
        run_path_with_cstr(&file, &|file| File::open_c(dir.as_fd(), file, opts))
    }

    pub fn open_c(fd: BorrowedFd<'_>, path: &CStr, opts: &OpenOptions) -> io::Result<File> {
        #[cfg(target_env = "p1")]
        if opts.use_wasip1 {
            let fd = unsafe {
                let fd = wasi::path_open(
                    fd.as_raw_fd() as wasi::Fd,
                    opts.wasip1_dirflags.unwrap_or(wasi::LOOKUPFLAGS_SYMLINK_FOLLOW),
                    path.to_str().map_err(|_| io::ErrorKind::InvalidInput)?,
                    opts.wasip1_oflags(),
                    opts.wasip1_rights_base(),
                    opts.wasip1_rights_inheriting(),
                    opts.wasip1_fdflags,
                )
                .map_err(crate::sys::err2io)?;
                WasiFd::from_raw_fd(fd as libc::c_int)
            };
            return Ok(File { fd });
        }
        c::openat(fd, path, opts.open_flags()?, Some(opts.open_mode()))
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        c::fstat(self.as_fd())
    }

    pub fn fsync(&self) -> io::Result<()> {
        c::fsync(self.as_fd())
    }

    pub fn datasync(&self) -> io::Result<()> {
        c::fdatasync(self.as_fd())
    }

    pub fn lock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn lock_shared(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn try_lock(&self) -> Result<(), TryLockError> {
        Err(TryLockError::Error(unsupported_err()))
    }

    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        Err(TryLockError::Error(unsupported_err()))
    }

    pub fn unlock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        let size = size.try_into().map_err(|_| io::ErrorKind::InvalidInput)?;
        c::ftruncate(self.as_fd(), size)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_vectored(&mut [IoSliceMut::new(buf)])
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.fd.read(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        true
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.fd.read_buf(cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.write_vectored(&[IoSlice::new(buf)])
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.fd.write(bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        true
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            // Casting to `i64` is fine, too large values will end up as
            // negative which will cause an error in `lseek64`.
            SeekFrom::Start(off) => (libc::SEEK_SET, off as i64),
            SeekFrom::End(off) => (libc::SEEK_END, off),
            SeekFrom::Current(off) => (libc::SEEK_CUR, off),
        };
        c::lseek(self.as_fd(), pos, whence)
    }

    pub fn size(&self) -> Option<io::Result<u64>> {
        None
    }

    pub fn tell(&self) -> io::Result<u64> {
        self.seek(SeekFrom::Current(0))
    }

    pub fn duplicate(&self) -> io::Result<File> {
        // https://github.com/CraneStation/wasmtime/blob/master/docs/WASI-rationale.md#why-no-dup
        unsupported()
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        // Permissions haven't been fully figured out in wasi yet, so this is
        // likely temporary
        unsupported()
    }

    pub fn set_times(&self, times: FileTimes) -> io::Result<()> {
        let omit = libc::timespec { tv_sec: 0, tv_nsec: libc::UTIME_OMIT };
        let times = [
            times.accessed.map(|t| t.to_timespec()).transpose()?.unwrap_or(omit),
            times.modified.map(|t| t.to_timespec()).transpose()?.unwrap_or(omit),
        ];
        c::futimens(self.as_fd(), &times)
    }

    #[cfg(target_env = "p1")]
    pub fn metadata_at(&self, flags: i32, path: &Path) -> io::Result<FileAttr> {
        run_path_with_cstr(path, &|path| c::fstatat(self.as_fd(), path, flags))
    }

    #[cfg(target_env = "p1")]
    pub fn readlink_at(&self, path: &Path) -> io::Result<PathBuf> {
        run_path_with_cstr(path, &|path| read_link(self.as_fd(), path))
    }

    #[cfg(target_env = "p1")]
    pub fn open_at(&self, path: &Path, opts: &OpenOptions) -> io::Result<File> {
        run_path_with_cstr(path, &|path| File::open_c(self.as_fd(), path, opts))
    }
}

impl AsInner<WasiFd> for File {
    #[inline]
    fn as_inner(&self) -> &WasiFd {
        &self.fd
    }
}

impl IntoInner<WasiFd> for File {
    fn into_inner(self) -> WasiFd {
        self.fd
    }
}

impl FromInner<WasiFd> for File {
    fn from_inner(fd: WasiFd) -> File {
        File { fd }
    }
}

impl AsFd for File {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl AsRawFd for File {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

impl IntoRawFd for File {
    fn into_raw_fd(self) -> RawFd {
        self.fd.into_raw_fd()
    }
}

impl FromRawFd for File {
    unsafe fn from_raw_fd(raw_fd: RawFd) -> Self {
        unsafe { Self { fd: FromRawFd::from_raw_fd(raw_fd) } }
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        let (dir, path) = open_parent(p)?;
        run_path_with_cstr(&path, &|path| c::mkdirat(dir.as_fd(), path))
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File").field("fd", &self.as_raw_fd()).finish()
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let mut opts = OpenOptions::new();
    opts.custom_flags(libc::O_DIRECTORY);
    opts.read(true);
    let dir = File::open(p, &opts)?;
    ReadDir::new(dir, p.to_path_buf())
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let (dir, file) = open_parent(p)?;
    run_path_with_cstr(&file, &|file| c::unlinkat(dir.as_fd(), file, 0))
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let (old, old_file) = open_parent(old)?;
    let (new, new_file) = open_parent(new)?;
    run_path_with_cstr(&old_file, &|old_file| {
        run_path_with_cstr(&new_file, &|new_file| {
            c::renameat(old.as_fd(), old_file, new.as_fd(), new_file)
        })
    })?;
    Ok(())
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    // Permissions haven't been fully figured out in wasi yet, so this is
    // likely temporary
    unsupported()
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let (dir, path) = open_parent(p)?;
    run_path_with_cstr(&path, &|path| c::unlinkat(dir.as_fd(), path, libc::AT_REMOVEDIR))
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    let (dir, file) = open_parent(p)?;
    run_path_with_cstr(&file, &|file| read_link(dir.as_fd(), file))
}

fn read_link(fd: BorrowedFd<'_>, path: &CStr) -> io::Result<PathBuf> {
    // Try to get a best effort initial capacity for the vector we're going to
    // fill. Note that if it's not a symlink we don't use a file to avoid
    // allocating gigabytes if you read_link a huge movie file by accident.
    // Additionally we add 1 to the initial size so if it doesn't change until
    // when we call `readlink` the returned length will be less than the
    // capacity, guaranteeing that we got all the data.
    let meta = c::fstatat(fd, path, libc::AT_SYMLINK_NOFOLLOW)?;
    let initial_size = if meta.file_type().is_symlink() {
        (meta.size() as usize).saturating_add(1)
    } else {
        1 // this'll fail in just a moment
    };

    // Now that we have an initial guess of how big to make our buffer, call
    // `readlink` in a loop until it fails or reports it filled fewer bytes than
    // we asked for, indicating we got everything.
    let mut destination = vec![0u8; initial_size];
    loop {
        let len = c::readlinkat(fd, path, &mut destination)?;
        if len < destination.len() {
            destination.truncate(len);
            destination.shrink_to_fit();
            return Ok(PathBuf::from(OsString::from_vec(destination)));
        }
        let amt_to_add = destination.len();
        destination.extend(iter::repeat(0).take(amt_to_add));
    }
}

pub fn symlink(original: &Path, link: &Path) -> io::Result<()> {
    let (link, link_file) = open_parent(link)?;
    run_path_with_cstr(&original, &|original| {
        run_path_with_cstr(&link_file, &|link_file| c::symlinkat(original, link.as_fd(), link_file))
    })?;
    Ok(())
}

pub fn link(original: &Path, link: &Path) -> io::Result<()> {
    let (original, original_file) = open_parent(original)?;
    let (link, link_file) = open_parent(link)?;

    run_path_with_cstr(&original_file, &|original_file| {
        run_path_with_cstr(&link_file, &|link_file| {
            c::linkat(
                original.as_fd(),
                original_file,
                link.as_fd(),
                link_file,
                // Pass 0 as the flags argument, meaning don't follow
                // symlinks.
                0,
            )
        })
    })?;
    Ok(())
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let (dir, file) = open_parent(p)?;
    run_path_with_cstr(&file, &|file| c::fstatat(dir.as_fd(), file, libc::AT_SYMLINK_FOLLOW))
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    let (dir, file) = open_parent(p)?;
    run_path_with_cstr(&file, &|file| c::fstatat(dir.as_fd(), file, libc::AT_SYMLINK_NOFOLLOW))
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    // This seems to not be in wasi's API yet, and we may need to end up
    // emulating it ourselves. For now just return an error.
    unsupported()
}

/// Attempts to open a bare path `p`.
///
/// WASI has no fundamental capability to do this. All syscalls and operations
/// are relative to already-open file descriptors. The C library, however,
/// manages a map of pre-opened file descriptors to their path, and then the C
/// library provides an API to look at this. In other words, when you want to
/// open a path `p`, you have to find a previously opened file descriptor in a
/// global table and then see if `p` is relative to that file descriptor.
///
/// This function, if successful, will return two items:
///
/// * The first is a `ManuallyDrop<WasiFd>`. This represents a pre-opened file
///   descriptor which we don't have ownership of, but we can use. You shouldn't
///   actually drop the `fd`.
///
/// * The second is a path that should be a part of `p` and represents a
///   relative traversal from the file descriptor specified to the desired
///   location `p`.
///
/// If successful you can use the returned file descriptor to perform
/// file-descriptor-relative operations on the path returned as well. The
/// `rights` argument indicates what operations are desired on the returned file
/// descriptor, and if successful the returned file descriptor should have the
/// appropriate rights for performing `rights` actions.
///
/// Note that this can fail if `p` doesn't look like it can be opened relative
/// to any pre-opened file descriptor.
fn open_parent(p: &Path) -> io::Result<(ManuallyDrop<WasiFd>, PathBuf)> {
    run_path_with_cstr(p, &|p| {
        let mut buf = Vec::<u8>::with_capacity(512);
        loop {
            unsafe {
                let mut relative_path = buf.as_mut_ptr().cast();
                let mut abs_prefix = ptr::null();
                let fd = libc::__wasilibc_find_relpath(
                    p.as_ptr(),
                    &mut abs_prefix,
                    &mut relative_path,
                    buf.capacity(),
                );
                if fd == -1 {
                    if io::Error::last_os_error().raw_os_error() == Some(libc::ENOMEM) {
                        // Trigger the internal buffer resizing logic of `Vec` by requiring
                        // more space than the current capacity.
                        let cap = buf.capacity();
                        buf.set_len(cap);
                        buf.reserve(1);
                        continue;
                    }
                    let msg = format!(
                        "failed to find a pre-opened file descriptor \
                        through which {p:?} could be opened",
                    );
                    return Err(io::Error::new(io::ErrorKind::Uncategorized, msg));
                }
                let relative = CStr::from_ptr(relative_path).to_bytes().to_vec();

                return Ok((
                    ManuallyDrop::new(WasiFd::from_raw_fd(fd as c_int)),
                    PathBuf::from(OsString::from_vec(relative)),
                ));
            }
        }
    })
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    use crate::fs::File;

    let mut reader = File::open(from)?;
    let mut writer = File::create(to)?;

    io::copy(&mut reader, &mut writer)
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let (parent, path) = open_parent(path)?;

    run_path_with_cstr(&path, &|path| remove_dir_all_recursive(parent.as_fd(), path))
}

fn remove_dir_all_recursive(parent: BorrowedFd<'_>, path: &CStr) -> io::Result<()> {
    // Open up a file descriptor for the directory itself. Note that we don't
    // follow symlinks here and we specifically open directories.
    //
    // At the root invocation of this function this will correctly handle
    // symlinks passed to the top-level `remove_dir_all`. At the recursive
    // level this will double-check that after the `readdir` call deduced this
    // was a directory it's still a directory by the time we open it up.
    //
    // If the opened file was actually a symlink then the symlink is deleted,
    // not the directory recursively.
    let fd = c::openat(parent, path, libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW, None)?;
    if fd.file_attr()?.file_type().is_symlink() {
        return c::unlinkat(parent.as_fd(), path, 0);
    }

    // this "root" is only used by `DirEntry::path` which we don't use below so
    // it's ok for this to be a bogus value
    let dummy_root = PathBuf::new();

    // Iterate over all the entries in this directory, and travel recursively if
    // necessary
    //
    // Note that all directory entries for this directory are read first before
    // any removal is done. This works around the fact that the WASIp1 API for
    // reading directories is not well-designed for handling mutations between
    // invocations of reading a directory. By reading all the entries at once
    // this ensures that, at least without concurrent modifications, it should
    // be possible to delete everything.
    for entry in ReadDir::new(fd, dummy_root)?.collect::<Vec<_>>() {
        let entry = entry?;
        let result: io::Result<()> = try {
            if entry.file_type()?.is_dir() {
                remove_dir_all_recursive(entry.dir_fd(), &entry.name)?;
            } else {
                c::unlinkat(entry.dir_fd(), &entry.name, 0)?;
            }
        };
        // ignore internal NotFound errors
        if let Err(err) = &result
            && err.kind() != io::ErrorKind::NotFound
        {
            return result;
        }
    }

    // Once all this directory's contents are deleted it should be safe to
    // delete the directory tiself.
    ignore_notfound(c::unlinkat(parent, path, libc::AT_REMOVEDIR))
}

mod c {
    use super::{File, FileAttr};
    use crate::ffi::CStr;
    use crate::io;
    use crate::mem::MaybeUninit;
    use crate::os::fd::{FromRawFd, IntoRawFd};
    use crate::os::wasi::io::{AsRawFd, BorrowedFd};
    use crate::sys::os::{cvt, errno};

    pub fn ftruncate(fd: BorrowedFd<'_>, size: libc::off_t) -> io::Result<()> {
        unsafe {
            cvt(libc::ftruncate(fd.as_raw_fd(), size))?;
        }
        Ok(())
    }

    pub fn linkat(
        oldfd: BorrowedFd<'_>,
        oldpath: &CStr,
        newfd: BorrowedFd<'_>,
        newpath: &CStr,
        flags: libc::c_int,
    ) -> io::Result<()> {
        cvt(unsafe {
            libc::linkat(
                oldfd.as_raw_fd(),
                oldpath.as_ptr(),
                newfd.as_raw_fd(),
                newpath.as_ptr(),
                flags,
            )
        })?;
        Ok(())
    }

    pub fn symlinkat(original: &CStr, newfd: BorrowedFd<'_>, newpath: &CStr) -> io::Result<()> {
        cvt(unsafe { libc::symlinkat(original.as_ptr(), newfd.as_raw_fd(), newpath.as_ptr()) })?;
        Ok(())
    }

    pub fn renameat(
        oldfd: BorrowedFd<'_>,
        oldpath: &CStr,
        newfd: BorrowedFd<'_>,
        newpath: &CStr,
    ) -> io::Result<()> {
        cvt(unsafe {
            libc::renameat(oldfd.as_raw_fd(), oldpath.as_ptr(), newfd.as_raw_fd(), newpath.as_ptr())
        })?;
        Ok(())
    }

    pub fn mkdirat(fd: BorrowedFd<'_>, path: &CStr) -> io::Result<()> {
        cvt(unsafe { libc::mkdirat(fd.as_raw_fd(), path.as_ptr(), 0o777) })?;
        Ok(())
    }

    pub fn futimens(fd: BorrowedFd<'_>, times: &[libc::timespec; 2]) -> io::Result<()> {
        unsafe {
            cvt(libc::futimens(fd.as_raw_fd(), times.as_ptr()))?;
        }
        Ok(())
    }

    pub fn lseek(fd: BorrowedFd<'_>, pos: i64, whence: libc::c_int) -> io::Result<u64> {
        let n = cvt(unsafe { libc::lseek(fd.as_raw_fd(), pos, whence) })?;
        Ok(n as u64)
    }

    pub fn fsync(fd: BorrowedFd<'_>) -> io::Result<()> {
        unsafe {
            cvt(libc::fsync(fd.as_raw_fd()))?;
        }
        Ok(())
    }

    pub fn fdatasync(fd: BorrowedFd<'_>) -> io::Result<()> {
        unsafe {
            cvt(libc::fdatasync(fd.as_raw_fd()))?;
        }
        Ok(())
    }

    pub fn fstat(fd: BorrowedFd<'_>) -> io::Result<FileAttr> {
        let mut stat = MaybeUninit::uninit();
        unsafe {
            cvt(libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()))?;
            Ok(FileAttr { stat: stat.assume_init() })
        }
    }

    pub fn fstatat(fd: BorrowedFd<'_>, path: &CStr, flags: libc::c_int) -> io::Result<FileAttr> {
        let mut stat = MaybeUninit::uninit();
        unsafe {
            cvt(libc::fstatat(fd.as_raw_fd(), path.as_ptr(), stat.as_mut_ptr(), flags))?;
            Ok(FileAttr { stat: stat.assume_init() })
        }
    }

    pub fn readlinkat(fd: BorrowedFd<'_>, path: &CStr, buf: &mut [u8]) -> io::Result<usize> {
        let len = cvt(unsafe {
            libc::readlinkat(fd.as_raw_fd(), path.as_ptr(), buf.as_mut_ptr().cast(), buf.len())
        })?;
        Ok(len as usize)
    }

    pub fn unlinkat(fd: BorrowedFd<'_>, path: &CStr, flags: libc::c_int) -> io::Result<()> {
        cvt(unsafe { libc::unlinkat(fd.as_raw_fd(), path.as_ptr(), flags) })?;
        Ok(())
    }

    pub fn openat(
        fd: BorrowedFd<'_>,
        path: &CStr,
        flags: libc::c_int,
        mode: Option<libc::c_int>,
    ) -> io::Result<File> {
        unsafe {
            let fd = match mode {
                Some(mode) => cvt(libc::openat(fd.as_raw_fd(), path.as_ptr(), flags, mode))?,
                None => cvt(libc::openat(fd.as_raw_fd(), path.as_ptr(), flags))?,
            };
            Ok(File::from_raw_fd(fd))
        }
    }

    pub struct Dirp {
        ptr: *mut libc::DIR,
    }

    impl Dirp {
        pub fn new(dir: File) -> io::Result<Dirp> {
            unsafe {
                let ptr = libc::fdopendir(dir.as_raw_fd());
                if ptr.is_null() {
                    return Err(io::Error::last_os_error());
                }
                let _ = dir.into_raw_fd(); // `ptr` now owns the fd
                Ok(Dirp { ptr })
            }
        }

        pub fn readdir(&self) -> io::Result<Option<*mut libc::dirent>> {
            unsafe {
                let entry = libc::readdir(self.ptr);
                if entry.is_null() {
                    let e = errno();
                    if e == 0 { Ok(None) } else { Err(io::Error::from_raw_os_error(e)) }
                } else {
                    Ok(Some(entry))
                }
            }
        }

        pub fn as_fd(&self) -> BorrowedFd<'_> {
            unsafe { BorrowedFd::borrow_raw(libc::dirfd(self.ptr)) }
        }
    }

    unsafe impl Send for Dirp {}
    unsafe impl Sync for Dirp {}

    impl Drop for Dirp {
        fn drop(&mut self) {
            unsafe {
                libc::closedir(self.ptr);
            }
        }
    }
}
