use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut, SeekFrom};
use crate::iter;
use crate::mem::{self, ManuallyDrop};
use crate::os::wasi::ffi::{OsStrExt, OsStringExt};
use crate::path::{Path, PathBuf};
use crate::ptr;
use crate::sync::Arc;
use crate::sys::fd::{DirCookie, WasiFd};
use crate::sys::time::SystemTime;
use crate::sys::unsupported;
use crate::sys_common::FromInner;

pub use crate::sys_common::fs::copy;
pub use crate::sys_common::fs::remove_dir_all;

pub struct File {
    fd: WasiFd,
}

#[derive(Clone)]
pub struct FileAttr {
    meta: libc::__wasi_filestat_t,
}

pub struct ReadDir {
    inner: Arc<ReadDirInner>,
    cookie: Option<DirCookie>,
    buf: Vec<u8>,
    offset: usize,
    cap: usize,
}

struct ReadDirInner {
    root: PathBuf,
    dir: File,
}

pub struct DirEntry {
    meta: libc::__wasi_dirent_t,
    name: Vec<u8>,
    inner: Arc<ReadDirInner>,
}

#[derive(Clone, Debug, Default)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    dirflags: libc::__wasi_lookupflags_t,
    fdflags: libc::__wasi_fdflags_t,
    oflags: libc::__wasi_oflags_t,
    rights_base: Option<libc::__wasi_rights_t>,
    rights_inheriting: Option<libc::__wasi_rights_t>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    readonly: bool,
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FileType {
    bits: libc::__wasi_filetype_t,
}

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    fn zero() -> FileAttr {
        FileAttr {
            meta: unsafe { mem::zeroed() },
        }
    }

    pub fn size(&self) -> u64 {
        self.meta.st_size
    }

    pub fn perm(&self) -> FilePermissions {
        // not currently implemented in wasi yet
        FilePermissions { readonly: false }
    }

    pub fn file_type(&self) -> FileType {
        FileType {
            bits: self.meta.st_filetype,
        }
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_wasi_timestamp(self.meta.st_mtim))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_wasi_timestamp(self.meta.st_atim))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_wasi_timestamp(self.meta.st_ctim))
    }

    pub fn as_wasi(&self) -> &libc::__wasi_filestat_t {
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

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.bits == libc::__WASI_FILETYPE_DIRECTORY
    }

    pub fn is_file(&self) -> bool {
        self.bits == libc::__WASI_FILETYPE_REGULAR_FILE
    }

    pub fn is_symlink(&self) -> bool {
        self.bits == libc::__WASI_FILETYPE_SYMBOLIC_LINK
    }

    pub fn bits(&self) -> libc::__wasi_filetype_t {
        self.bits
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadDir").finish()
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        loop {
            // If we've reached the capacity of our buffer then we need to read
            // some more from the OS, otherwise we pick up at our old offset.
            let offset = if self.offset == self.cap {
                let cookie = self.cookie.take()?;
                match self.inner.dir.fd.readdir(&mut self.buf, cookie) {
                    Ok(bytes) => self.cap = bytes,
                    Err(e) => return Some(Err(e)),
                }
                self.offset = 0;
                self.cookie = Some(cookie);

                // If we didn't actually read anything, this is in theory the
                // end of the directory.
                if self.cap == 0 {
                    self.cookie = None;
                    return None;
                }

                0
            } else {
                self.offset
            };
            let data = &self.buf[offset..self.cap];

            // If we're not able to read a directory entry then that means it
            // must have been truncated at the end of the buffer, so reset our
            // offset so we can go back and reread into the buffer, picking up
            // where we last left off.
            let dirent_size = mem::size_of::<libc::__wasi_dirent_t>();
            if data.len() < dirent_size {
                assert!(self.cookie.is_some());
                assert!(self.buf.len() >= dirent_size);
                self.offset = self.cap;
                continue;
            }
            let (dirent, data) = data.split_at(dirent_size);
            let dirent =
                unsafe { ptr::read_unaligned(dirent.as_ptr() as *const libc::__wasi_dirent_t) };

            // If the file name was truncated, then we need to reinvoke
            // `readdir` so we truncate our buffer to start over and reread this
            // descriptor. Note that if our offset is 0 that means the file name
            // is massive and we need a bigger buffer.
            if data.len() < dirent.d_namlen as usize {
                if offset == 0 {
                    let amt_to_add = self.buf.capacity();
                    self.buf.extend(iter::repeat(0).take(amt_to_add));
                }
                assert!(self.cookie.is_some());
                self.offset = self.cap;
                continue;
            }
            self.cookie = Some(dirent.d_next);
            self.offset = offset + dirent_size + dirent.d_namlen as usize;

            let name = &data[..(dirent.d_namlen as usize)];

            // These names are skipped on all other platforms, so let's skip
            // them here too
            if name == b"." || name == b".." {
                continue;
            }

            return Some(Ok(DirEntry {
                meta: dirent,
                name: name.to_vec(),
                inner: self.inner.clone(),
            }));
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
        metadata_at(
            &self.inner.dir.fd,
            0,
            OsStr::from_bytes(&self.name).as_ref(),
        )
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(FileType {
            bits: self.meta.d_type,
        })
    }

    pub fn ino(&self) -> libc::__wasi_inode_t {
        self.meta.d_ino
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        let mut base = OpenOptions::default();
        base.dirflags = libc::__WASI_LOOKUP_SYMLINK_FOLLOW;
        return base;
    }

    pub fn read(&mut self, read: bool) {
        self.read = read;
    }

    pub fn write(&mut self, write: bool) {
        self.write = write;
    }

    pub fn truncate(&mut self, truncate: bool) {
        self.oflag(libc::__WASI_O_TRUNC, truncate);
    }

    pub fn create(&mut self, create: bool) {
        self.oflag(libc::__WASI_O_CREAT, create);
    }

    pub fn create_new(&mut self, create_new: bool) {
        self.oflag(libc::__WASI_O_EXCL, create_new);
        self.oflag(libc::__WASI_O_CREAT, create_new);
    }

    pub fn directory(&mut self, directory: bool) {
        self.oflag(libc::__WASI_O_DIRECTORY, directory);
    }

    fn oflag(&mut self, bit: libc::__wasi_oflags_t, set: bool) {
        if set {
            self.oflags |= bit;
        } else {
            self.oflags &= !bit;
        }
    }

    pub fn append(&mut self, set: bool) {
        self.fdflag(libc::__WASI_FDFLAG_APPEND, set);
    }

    pub fn dsync(&mut self, set: bool) {
        self.fdflag(libc::__WASI_FDFLAG_DSYNC, set);
    }

    pub fn nonblock(&mut self, set: bool) {
        self.fdflag(libc::__WASI_FDFLAG_NONBLOCK, set);
    }

    pub fn rsync(&mut self, set: bool) {
        self.fdflag(libc::__WASI_FDFLAG_RSYNC, set);
    }

    pub fn sync(&mut self, set: bool) {
        self.fdflag(libc::__WASI_FDFLAG_SYNC, set);
    }

    fn fdflag(&mut self, bit: libc::__wasi_fdflags_t, set: bool) {
        if set {
            self.fdflags |= bit;
        } else {
            self.fdflags &= !bit;
        }
    }

    pub fn fs_rights_base(&mut self, rights: libc::__wasi_rights_t) {
        self.rights_base = Some(rights);
    }

    pub fn fs_rights_inheriting(&mut self, rights: libc::__wasi_rights_t) {
        self.rights_inheriting = Some(rights);
    }

    fn rights_base(&self) -> libc::__wasi_rights_t {
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
            base |= libc::__WASI_RIGHT_FD_READ;
            base |= libc::__WASI_RIGHT_FD_READDIR;
        }
        if self.write {
            base |= libc::__WASI_RIGHT_FD_WRITE;
            base |= libc::__WASI_RIGHT_FD_DATASYNC;
            base |= libc::__WASI_RIGHT_FD_ALLOCATE;
            base |= libc::__WASI_RIGHT_FD_FILESTAT_SET_SIZE;
        }

        // FIXME: some of these should probably be read-only or write-only...
        base |= libc::__WASI_RIGHT_FD_ADVISE;
        base |= libc::__WASI_RIGHT_FD_FDSTAT_SET_FLAGS;
        base |= libc::__WASI_RIGHT_FD_FILESTAT_SET_TIMES;
        base |= libc::__WASI_RIGHT_FD_SEEK;
        base |= libc::__WASI_RIGHT_FD_SYNC;
        base |= libc::__WASI_RIGHT_FD_TELL;
        base |= libc::__WASI_RIGHT_PATH_CREATE_DIRECTORY;
        base |= libc::__WASI_RIGHT_PATH_CREATE_FILE;
        base |= libc::__WASI_RIGHT_PATH_FILESTAT_GET;
        base |= libc::__WASI_RIGHT_PATH_LINK_SOURCE;
        base |= libc::__WASI_RIGHT_PATH_LINK_TARGET;
        base |= libc::__WASI_RIGHT_PATH_OPEN;
        base |= libc::__WASI_RIGHT_PATH_READLINK;
        base |= libc::__WASI_RIGHT_PATH_REMOVE_DIRECTORY;
        base |= libc::__WASI_RIGHT_PATH_RENAME_SOURCE;
        base |= libc::__WASI_RIGHT_PATH_RENAME_TARGET;
        base |= libc::__WASI_RIGHT_PATH_SYMLINK;
        base |= libc::__WASI_RIGHT_PATH_UNLINK_FILE;
        base |= libc::__WASI_RIGHT_POLL_FD_READWRITE;

        return base;
    }

    fn rights_inheriting(&self) -> libc::__wasi_rights_t {
        self.rights_inheriting.unwrap_or_else(|| self.rights_base())
    }

    pub fn lookup_flags(&mut self, flags: libc::__wasi_lookupflags_t) {
        self.dirflags = flags;
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let (dir, file) = open_parent(path, libc::__WASI_RIGHT_PATH_OPEN)?;
        open_at(&dir, &file, opts)
    }

    pub fn open_at(&self, path: &Path, opts: &OpenOptions) -> io::Result<File> {
        open_at(&self.fd, path, opts)
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let mut ret = FileAttr::zero();
        self.fd.filestat_get(&mut ret.meta)?;
        Ok(ret)
    }

    pub fn metadata_at(
        &self,
        flags: libc::__wasi_lookupflags_t,
        path: &Path,
    ) -> io::Result<FileAttr> {
        metadata_at(&self.fd, flags, path)
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.fd.sync()
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fd.datasync()
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

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.write_vectored(&[IoSlice::new(buf)])
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.fd.write(bufs)
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

    pub fn fd(&self) -> &WasiFd {
        &self.fd
    }

    pub fn into_fd(self) -> WasiFd {
        self.fd
    }

    pub fn read_link(&self, file: &Path) -> io::Result<PathBuf> {
        read_link(&self.fd, file)
    }
}

impl FromInner<u32> for File {
    fn from_inner(fd: u32) -> File {
        unsafe {
            File {
                fd: WasiFd::from_raw(fd),
            }
        }
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        let (dir, file) = open_parent(p, libc::__WASI_RIGHT_PATH_CREATE_DIRECTORY)?;
        dir.create_directory(file.as_os_str().as_bytes())
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File")
            .field("fd", &self.fd.as_raw())
            .finish()
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let mut opts = OpenOptions::new();
    opts.directory(true);
    opts.read(true);
    let dir = File::open(p, &opts)?;
    Ok(ReadDir {
        cookie: Some(0),
        buf: vec![0; 128],
        offset: 0,
        cap: 0,
        inner: Arc::new(ReadDirInner {
            dir,
            root: p.to_path_buf(),
        }),
    })
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let (dir, file) = open_parent(p, libc::__WASI_RIGHT_PATH_UNLINK_FILE)?;
    dir.unlink_file(file.as_os_str().as_bytes())
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let (old, old_file) = open_parent(old, libc::__WASI_RIGHT_PATH_RENAME_SOURCE)?;
    let (new, new_file) = open_parent(new, libc::__WASI_RIGHT_PATH_RENAME_TARGET)?;
    old.rename(
        old_file.as_os_str().as_bytes(),
        &new,
        new_file.as_os_str().as_bytes(),
    )
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    // Permissions haven't been fully figured out in wasi yet, so this is
    // likely temporary
    unsupported()
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let (dir, file) = open_parent(p, libc::__WASI_RIGHT_PATH_REMOVE_DIRECTORY)?;
    dir.remove_directory(file.as_os_str().as_bytes())
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    let (dir, file) = open_parent(p, libc::__WASI_RIGHT_PATH_READLINK)?;
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
    let file = file.as_os_str().as_bytes();
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

pub fn symlink(src: &Path, dst: &Path) -> io::Result<()> {
    let (dst, dst_file) = open_parent(dst, libc::__WASI_RIGHT_PATH_SYMLINK)?;
    dst.symlink(src.as_os_str().as_bytes(), dst_file.as_os_str().as_bytes())
}

pub fn link(src: &Path, dst: &Path) -> io::Result<()> {
    let (src, src_file) = open_parent(src, libc::__WASI_RIGHT_PATH_LINK_SOURCE)?;
    let (dst, dst_file) = open_parent(dst, libc::__WASI_RIGHT_PATH_LINK_TARGET)?;
    src.link(
        libc::__WASI_LOOKUP_SYMLINK_FOLLOW,
        src_file.as_os_str().as_bytes(),
        &dst,
        dst_file.as_os_str().as_bytes(),
    )
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let (dir, file) = open_parent(p, libc::__WASI_RIGHT_PATH_FILESTAT_GET)?;
    metadata_at(&dir, libc::__WASI_LOOKUP_SYMLINK_FOLLOW, &file)
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    let (dir, file) = open_parent(p, libc::__WASI_RIGHT_PATH_FILESTAT_GET)?;
    metadata_at(&dir, 0, &file)
}

fn metadata_at(
    fd: &WasiFd,
    flags: libc::__wasi_lookupflags_t,
    path: &Path,
) -> io::Result<FileAttr> {
    let mut ret = FileAttr::zero();
    fd.path_filestat_get(flags, path.as_os_str().as_bytes(), &mut ret.meta)?;
    Ok(ret)
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    // This seems to not be in wasi's API yet, and we may need to end up
    // emulating it ourselves. For now just return an error.
    unsupported()
}

fn open_at(fd: &WasiFd, path: &Path, opts: &OpenOptions) -> io::Result<File> {
    let fd = fd.open(
        opts.dirflags,
        path.as_os_str().as_bytes(),
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
/// manages a map of preopened file descriptors to their path, and then the C
/// library provides an API to look at this. In other words, when you want to
/// open a path `p`, you have to find a previously opened file descriptor in a
/// global table and then see if `p` is relative to that file descriptor.
///
/// This function, if successful, will return two items:
///
/// * The first is a `ManuallyDrop<WasiFd>`. This represents a preopened file
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
/// to any preopened file descriptor.
fn open_parent(
    p: &Path,
    rights: libc::__wasi_rights_t,
) -> io::Result<(ManuallyDrop<WasiFd>, PathBuf)> {
    let p = CString::new(p.as_os_str().as_bytes())?;
    unsafe {
        let mut ret = ptr::null();
        let fd = __wasilibc_find_relpath(p.as_ptr(), rights, 0, &mut ret);
        if fd == -1 {
            let msg = format!(
                "failed to find a preopened file descriptor \
                 through which {:?} could be opened",
                p
            );
            return Err(io::Error::new(io::ErrorKind::Other, msg));
        }
        let path = Path::new(OsStr::from_bytes(CStr::from_ptr(ret).to_bytes()));

        // FIXME: right now `path` is a pointer into `p`, the `CString` above.
        // When we return `p` is deallocated and we can't use it, so we need to
        // currently separately allocate `path`. If this becomes an issue though
        // we should probably turn this into a closure-taking interface or take
        // `&CString` and then pass off `&Path` tied to the same lifetime.
        let path = path.to_path_buf();

        return Ok((ManuallyDrop::new(WasiFd::from_raw(fd as u32)), path));
    }

    // FIXME(rust-lang/libc#1314) use the `libc` crate for this when the API
    // there is published
    extern "C" {
        pub fn __wasilibc_find_relpath(
            path: *const libc::c_char,
            rights_base: libc::__wasi_rights_t,
            rights_inheriting: libc::__wasi_rights_t,
            relative_path: *mut *const libc::c_char,
        ) -> libc::c_int;
    }
}
