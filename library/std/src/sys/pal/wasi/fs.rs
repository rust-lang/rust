#![forbid(unsafe_op_in_unsafe_fn)]

use super::fd::WasiFd;
use crate::ffi::{CStr, OsStr, OsString};
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::mem::{self, ManuallyDrop};
use crate::os::raw::c_int;
use crate::os::wasi::ffi::{OsStrExt, OsStringExt};
use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
use crate::path::{Path, PathBuf};
use crate::sync::Arc;
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::time::SystemTime;
use crate::sys::unsupported;
pub use crate::sys_common::fs::exists;
use crate::sys_common::{AsInner, FromInner, IntoInner, ignore_notfound};
use crate::{fmt, iter, ptr};

pub struct File {
    fd: WasiFd,
}

#[derive(Clone)]
pub struct FileAttr {
    meta: wasi::Filestat,
}

pub struct ReadDir {
    inner: Arc<ReadDirInner>,
    state: ReadDirState,
}

enum ReadDirState {
    /// Fill `buf` with `buf.len()` bytes starting from `next_read_offset`.
    FillBuffer {
        next_read_offset: wasi::Dircookie,
        buf: Vec<u8>,
    },
    ProcessEntry {
        buf: Vec<u8>,
        next_read_offset: Option<wasi::Dircookie>,
        offset: usize,
    },
    /// There is no more data to get in [`Self::FillBuffer`]; keep returning
    /// entries via ProcessEntry until `buf` is exhausted.
    RunUntilExhaustion {
        buf: Vec<u8>,
        offset: usize,
    },
    Done,
}

struct ReadDirInner {
    root: PathBuf,
    dir: File,
}

pub struct DirEntry {
    meta: wasi::Dirent,
    name: Vec<u8>,
    inner: Arc<ReadDirInner>,
}

#[derive(Clone, Debug, Default)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    dirflags: wasi::Lookupflags,
    fdflags: wasi::Fdflags,
    oflags: wasi::Oflags,
    rights_base: Option<wasi::Rights>,
    rights_inheriting: Option<wasi::Rights>,
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
    bits: wasi::Filetype,
}

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.meta.size
    }

    pub fn perm(&self) -> FilePermissions {
        // not currently implemented in wasi yet
        FilePermissions { readonly: false }
    }

    pub fn file_type(&self) -> FileType {
        FileType { bits: self.meta.filetype }
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_wasi_timestamp(self.meta.mtim))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_wasi_timestamp(self.meta.atim))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_wasi_timestamp(self.meta.ctim))
    }

    pub(crate) fn as_wasi(&self) -> &wasi::Filestat {
        &self.meta
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
        self.bits == wasi::FILETYPE_DIRECTORY
    }

    pub fn is_file(&self) -> bool {
        self.bits == wasi::FILETYPE_REGULAR_FILE
    }

    pub fn is_symlink(&self) -> bool {
        self.bits == wasi::FILETYPE_SYMBOLIC_LINK
    }

    pub(crate) fn bits(&self) -> wasi::Filetype {
        self.bits
    }
}

impl ReadDir {
    fn new(dir: File, root: PathBuf) -> ReadDir {
        ReadDir {
            inner: Arc::new(ReadDirInner { dir, root }),
            state: ReadDirState::FillBuffer { next_read_offset: 0, buf: vec![0; 128] },
        }
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
        match &mut self.state {
            ReadDirState::FillBuffer { next_read_offset, buf } => {
                let result = self.inner.dir.fd.readdir(buf, *next_read_offset);
                match result {
                    Ok(read_bytes) => {
                        if read_bytes < buf.len() {
                            buf.truncate(read_bytes);
                            self.state =
                                ReadDirState::RunUntilExhaustion { buf: mem::take(buf), offset: 0 };
                        } else {
                            debug_assert_eq!(read_bytes, buf.len());
                            self.state = ReadDirState::ProcessEntry {
                                buf: mem::take(buf),
                                offset: 0,
                                next_read_offset: Some(*next_read_offset),
                            };
                        }
                        self.next()
                    }
                    Err(e) => {
                        self.state = ReadDirState::Done;
                        return Some(Err(e));
                    }
                }
            }
            ReadDirState::ProcessEntry { buf, next_read_offset, offset } => {
                let contents = &buf[*offset..];
                const DIRENT_SIZE: usize = crate::mem::size_of::<wasi::Dirent>();
                if contents.len() >= DIRENT_SIZE {
                    let (dirent, data) = contents.split_at(DIRENT_SIZE);
                    let dirent =
                        unsafe { ptr::read_unaligned(dirent.as_ptr() as *const wasi::Dirent) };
                    // If the file name was truncated, then we need to reinvoke
                    // `readdir` so we truncate our buffer to start over and reread this
                    // descriptor.
                    if data.len() < dirent.d_namlen as usize {
                        if buf.len() < dirent.d_namlen as usize + DIRENT_SIZE {
                            buf.resize(dirent.d_namlen as usize + DIRENT_SIZE, 0);
                        }
                        if let Some(next_read_offset) = *next_read_offset {
                            self.state =
                                ReadDirState::FillBuffer { next_read_offset, buf: mem::take(buf) };
                        } else {
                            self.state = ReadDirState::Done;
                        }

                        return self.next();
                    }
                    next_read_offset.as_mut().map(|cookie| {
                        *cookie = dirent.d_next;
                    });
                    *offset = *offset + DIRENT_SIZE + dirent.d_namlen as usize;

                    let name = &data[..(dirent.d_namlen as usize)];

                    // These names are skipped on all other platforms, so let's skip
                    // them here too
                    if name == b"." || name == b".." {
                        return self.next();
                    }

                    return Some(Ok(DirEntry {
                        meta: dirent,
                        name: name.to_vec(),
                        inner: self.inner.clone(),
                    }));
                } else if let Some(next_read_offset) = *next_read_offset {
                    self.state = ReadDirState::FillBuffer { next_read_offset, buf: mem::take(buf) };
                } else {
                    self.state = ReadDirState::Done;
                }
                self.next()
            }
            ReadDirState::RunUntilExhaustion { buf, offset } => {
                if *offset >= buf.len() {
                    self.state = ReadDirState::Done;
                } else {
                    self.state = ReadDirState::ProcessEntry {
                        buf: mem::take(buf),
                        offset: *offset,
                        next_read_offset: None,
                    };
                }

                self.next()
            }
            ReadDirState::Done => None,
        }
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        let name = OsStr::from_bytes(&self.name);
        self.inner.root.join(name)
    }

    pub fn file_name(&self) -> OsString {
        OsString::from_vec(self.name.clone())
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        metadata_at(&self.inner.dir.fd, 0, OsStr::from_bytes(&self.name).as_ref())
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(FileType { bits: self.meta.d_type })
    }

    pub fn ino(&self) -> wasi::Inode {
        self.meta.d_ino
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        let mut base = OpenOptions::default();
        base.dirflags = wasi::LOOKUPFLAGS_SYMLINK_FOLLOW;
        base
    }

    pub fn read(&mut self, read: bool) {
        self.read = read;
    }

    pub fn write(&mut self, write: bool) {
        self.write = write;
    }

    pub fn truncate(&mut self, truncate: bool) {
        self.oflag(wasi::OFLAGS_TRUNC, truncate);
    }

    pub fn create(&mut self, create: bool) {
        self.oflag(wasi::OFLAGS_CREAT, create);
    }

    pub fn create_new(&mut self, create_new: bool) {
        self.oflag(wasi::OFLAGS_EXCL, create_new);
        self.oflag(wasi::OFLAGS_CREAT, create_new);
    }

    pub fn directory(&mut self, directory: bool) {
        self.oflag(wasi::OFLAGS_DIRECTORY, directory);
    }

    fn oflag(&mut self, bit: wasi::Oflags, set: bool) {
        if set {
            self.oflags |= bit;
        } else {
            self.oflags &= !bit;
        }
    }

    pub fn append(&mut self, append: bool) {
        self.append = append;
        self.fdflag(wasi::FDFLAGS_APPEND, append);
    }

    pub fn dsync(&mut self, set: bool) {
        self.fdflag(wasi::FDFLAGS_DSYNC, set);
    }

    pub fn nonblock(&mut self, set: bool) {
        self.fdflag(wasi::FDFLAGS_NONBLOCK, set);
    }

    pub fn rsync(&mut self, set: bool) {
        self.fdflag(wasi::FDFLAGS_RSYNC, set);
    }

    pub fn sync(&mut self, set: bool) {
        self.fdflag(wasi::FDFLAGS_SYNC, set);
    }

    fn fdflag(&mut self, bit: wasi::Fdflags, set: bool) {
        if set {
            self.fdflags |= bit;
        } else {
            self.fdflags &= !bit;
        }
    }

    pub fn fs_rights_base(&mut self, rights: wasi::Rights) {
        self.rights_base = Some(rights);
    }

    pub fn fs_rights_inheriting(&mut self, rights: wasi::Rights) {
        self.rights_inheriting = Some(rights);
    }

    fn rights_base(&self) -> wasi::Rights {
        if let Some(rights) = self.rights_base {
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

    fn rights_inheriting(&self) -> wasi::Rights {
        self.rights_inheriting.unwrap_or_else(|| self.rights_base())
    }

    pub fn lookup_flags(&mut self, flags: wasi::Lookupflags) {
        self.dirflags = flags;
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let (dir, file) = open_parent(path)?;
        open_at(&dir, &file, opts)
    }

    pub fn open_at(&self, path: &Path, opts: &OpenOptions) -> io::Result<File> {
        open_at(&self.fd, path, opts)
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        self.fd.filestat_get().map(|meta| FileAttr { meta })
    }

    pub fn metadata_at(&self, flags: wasi::Lookupflags, path: &Path) -> io::Result<FileAttr> {
        metadata_at(&self.fd, flags, path)
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.fd.sync()
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fd.datasync()
    }

    pub fn lock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn lock_shared(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn try_lock(&self) -> io::Result<bool> {
        unsupported()
    }

    pub fn try_lock_shared(&self) -> io::Result<bool> {
        unsupported()
    }

    pub fn unlock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        self.fd.filestat_set_size(size)
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
        self.fd.seek(pos)
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
        let to_timestamp = |time: Option<SystemTime>| match time {
            Some(time) if let Some(ts) = time.to_wasi_timestamp() => Ok(ts),
            Some(_) => Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "timestamp is too large to set as a file time",
            )),
            None => Ok(0),
        };
        self.fd.filestat_set_times(
            to_timestamp(times.accessed)?,
            to_timestamp(times.modified)?,
            times.accessed.map_or(0, |_| wasi::FSTFLAGS_ATIM)
                | times.modified.map_or(0, |_| wasi::FSTFLAGS_MTIM),
        )
    }

    pub fn read_link(&self, file: &Path) -> io::Result<PathBuf> {
        read_link(&self.fd, file)
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
        let (dir, file) = open_parent(p)?;
        dir.create_directory(osstr2str(file.as_ref())?)
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File").field("fd", &self.as_raw_fd()).finish()
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let mut opts = OpenOptions::new();
    opts.directory(true);
    opts.read(true);
    let dir = File::open(p, &opts)?;
    Ok(ReadDir::new(dir, p.to_path_buf()))
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let (dir, file) = open_parent(p)?;
    dir.unlink_file(osstr2str(file.as_ref())?)
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let (old, old_file) = open_parent(old)?;
    let (new, new_file) = open_parent(new)?;
    old.rename(osstr2str(old_file.as_ref())?, &new, osstr2str(new_file.as_ref())?)
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    // Permissions haven't been fully figured out in wasi yet, so this is
    // likely temporary
    unsupported()
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let (dir, file) = open_parent(p)?;
    dir.remove_directory(osstr2str(file.as_ref())?)
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    let (dir, file) = open_parent(p)?;
    read_link(&dir, &file)
}

fn read_link(fd: &WasiFd, file: &Path) -> io::Result<PathBuf> {
    // Try to get a best effort initial capacity for the vector we're going to
    // fill. Note that if it's not a symlink we don't use a file to avoid
    // allocating gigabytes if you read_link a huge movie file by accident.
    // Additionally we add 1 to the initial size so if it doesn't change until
    // when we call `readlink` the returned length will be less than the
    // capacity, guaranteeing that we got all the data.
    let meta = metadata_at(fd, 0, file)?;
    let initial_size = if meta.file_type().is_symlink() {
        (meta.size() as usize).saturating_add(1)
    } else {
        1 // this'll fail in just a moment
    };

    // Now that we have an initial guess of how big to make our buffer, call
    // `readlink` in a loop until it fails or reports it filled fewer bytes than
    // we asked for, indicating we got everything.
    let file = osstr2str(file.as_ref())?;
    let mut destination = vec![0u8; initial_size];
    loop {
        let len = fd.readlink(file, &mut destination)?;
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
    link.symlink(osstr2str(original.as_ref())?, osstr2str(link_file.as_ref())?)
}

pub fn link(original: &Path, link: &Path) -> io::Result<()> {
    let (original, original_file) = open_parent(original)?;
    let (link, link_file) = open_parent(link)?;
    // Pass 0 as the flags argument, meaning don't follow symlinks.
    original.link(0, osstr2str(original_file.as_ref())?, &link, osstr2str(link_file.as_ref())?)
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let (dir, file) = open_parent(p)?;
    metadata_at(&dir, wasi::LOOKUPFLAGS_SYMLINK_FOLLOW, &file)
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    let (dir, file) = open_parent(p)?;
    metadata_at(&dir, 0, &file)
}

fn metadata_at(fd: &WasiFd, flags: wasi::Lookupflags, path: &Path) -> io::Result<FileAttr> {
    let meta = fd.path_filestat_get(flags, osstr2str(path.as_ref())?)?;
    Ok(FileAttr { meta })
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    // This seems to not be in wasi's API yet, and we may need to end up
    // emulating it ourselves. For now just return an error.
    unsupported()
}

fn open_at(fd: &WasiFd, path: &Path, opts: &OpenOptions) -> io::Result<File> {
    let fd = fd.open(
        opts.dirflags,
        osstr2str(path.as_ref())?,
        opts.oflags,
        opts.rights_base(),
        opts.rights_inheriting(),
        opts.fdflags,
    )?;
    Ok(File { fd })
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
                let mut relative_path = buf.as_ptr().cast();
                let mut abs_prefix = ptr::null();
                let fd = __wasilibc_find_relpath(
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

        unsafe extern "C" {
            pub fn __wasilibc_find_relpath(
                path: *const libc::c_char,
                abs_prefix: *mut *const libc::c_char,
                relative_path: *mut *const libc::c_char,
                relative_path_len: libc::size_t,
            ) -> libc::c_int;
        }
    })
}

pub fn osstr2str(f: &OsStr) -> io::Result<&str> {
    f.to_str().ok_or_else(|| io::const_error!(io::ErrorKind::Uncategorized, "input must be utf-8"))
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    use crate::fs::File;

    let mut reader = File::open(from)?;
    let mut writer = File::create(to)?;

    io::copy(&mut reader, &mut writer)
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let (parent, path) = open_parent(path)?;
    remove_dir_all_recursive(&parent, &path)
}

fn remove_dir_all_recursive(parent: &WasiFd, path: &Path) -> io::Result<()> {
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
    let mut opts = OpenOptions::new();
    opts.lookup_flags(0);
    opts.directory(true);
    opts.read(true);
    let fd = open_at(parent, path, &opts)?;
    if fd.file_attr()?.file_type().is_symlink() {
        return parent.unlink_file(osstr2str(path.as_ref())?);
    }

    // this "root" is only used by `DirEntry::path` which we don't use below so
    // it's ok for this to be a bogus value
    let dummy_root = PathBuf::new();

    // Iterate over all the entries in this directory, and travel recursively if
    // necessary
    for entry in ReadDir::new(fd, dummy_root) {
        let entry = entry?;
        let path = crate::str::from_utf8(&entry.name).map_err(|_| {
            io::const_error!(io::ErrorKind::Uncategorized, "invalid utf-8 file name found")
        })?;

        let result: io::Result<()> = try {
            if entry.file_type()?.is_dir() {
                remove_dir_all_recursive(&entry.inner.dir.fd, path.as_ref())?;
            } else {
                entry.inner.dir.fd.unlink_file(path)?;
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
    ignore_notfound(parent.remove_directory(osstr2str(path.as_ref())?))
}
