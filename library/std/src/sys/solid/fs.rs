use super::{abi, error};
use crate::{
    ffi::{CStr, CString, OsStr, OsString},
    fmt,
    io::{self, IoSlice, IoSliceMut, SeekFrom},
    mem::MaybeUninit,
    os::raw::{c_int, c_short},
    os::solid::ffi::OsStrExt,
    path::{Path, PathBuf},
    sync::Arc,
    sys::time::SystemTime,
    sys::unsupported,
};

pub use crate::sys_common::fs::try_exists;

/// A file descriptor.
#[derive(Clone, Copy)]
#[rustc_layout_scalar_valid_range_start(0)]
// libstd/os/raw/mod.rs assures me that every libstd-supported platform has a
// 32-bit c_int. Below is -2, in two's complement, but that only works out
// because c_int is 32 bits.
#[rustc_layout_scalar_valid_range_end(0xFF_FF_FF_FE)]
struct FileDesc {
    fd: c_int,
}

impl FileDesc {
    #[inline]
    fn new(fd: c_int) -> FileDesc {
        assert_ne!(fd, -1i32);
        // Safety: we just asserted that the value is in the valid range and
        // isn't `-1` (the only value bigger than `0xFF_FF_FF_FE` unsigned)
        unsafe { FileDesc { fd } }
    }

    #[inline]
    fn raw(&self) -> c_int {
        self.fd
    }
}

pub struct File {
    fd: FileDesc,
}

#[derive(Clone)]
pub struct FileAttr {
    stat: abi::stat,
}

// all DirEntry's will have a reference to this struct
struct InnerReadDir {
    dirp: abi::S_DIR,
    root: PathBuf,
}

pub struct ReadDir {
    inner: Arc<InnerReadDir>,
}

pub struct DirEntry {
    entry: abi::dirent,
    inner: Arc<InnerReadDir>,
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    // generic
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    // system-specific
    custom_flags: i32,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions(c_short);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileType(c_short);

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.stat.st_size as u64
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions(self.stat.st_mode)
    }

    pub fn file_type(&self) -> FileType {
        FileType(self.stat.st_mode)
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_time_t(self.stat.st_mtime))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_time_t(self.stat.st_atime))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from_time_t(self.stat.st_ctime))
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        (self.0 & abi::S_IWRITE) == 0
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.0 &= !abi::S_IWRITE;
        } else {
            self.0 |= abi::S_IWRITE;
        }
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.is(abi::S_IFDIR)
    }
    pub fn is_file(&self) -> bool {
        self.is(abi::S_IFREG)
    }
    pub fn is_symlink(&self) -> bool {
        false
    }

    pub fn is(&self, mode: c_short) -> bool {
        self.0 & abi::S_IFMT == mode
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    unsafe {
        let mut dir = MaybeUninit::uninit();
        error::SolidError::err_if_negative(abi::SOLID_FS_OpenDir(
            cstr(p)?.as_ptr(),
            dir.as_mut_ptr(),
        ))
        .map_err(|e| e.as_io_error())?;
        let inner = Arc::new(InnerReadDir { dirp: dir.assume_init(), root: p.to_owned() });
        Ok(ReadDir { inner })
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This will only be called from std::fs::ReadDir, which will add a "ReadDir()" frame.
        // Thus the result will be e g 'ReadDir("/home")'
        fmt::Debug::fmt(&*self.inner.root, f)
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        unsafe {
            let mut out_dirent = MaybeUninit::uninit();
            error::SolidError::err_if_negative(abi::SOLID_FS_ReadDir(
                self.inner.dirp,
                out_dirent.as_mut_ptr(),
            ))
            .ok()?;
            Some(Ok(DirEntry { entry: out_dirent.assume_init(), inner: Arc::clone(&self.inner) }))
        }
    }
}

impl Drop for InnerReadDir {
    fn drop(&mut self) {
        unsafe { abi::SOLID_FS_CloseDir(self.dirp) };
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.inner.root.join(OsStr::from_bytes(
            unsafe { CStr::from_ptr(self.entry.d_name.as_ptr()) }.to_bytes(),
        ))
    }

    pub fn file_name(&self) -> OsString {
        OsStr::from_bytes(unsafe { CStr::from_ptr(self.entry.d_name.as_ptr()) }.to_bytes())
            .to_os_string()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        lstat(&self.path())
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        match self.entry.d_type {
            abi::DT_CHR => Ok(FileType(abi::S_IFCHR)),
            abi::DT_FIFO => Ok(FileType(abi::S_IFIFO)),
            abi::DT_REG => Ok(FileType(abi::S_IFREG)),
            abi::DT_DIR => Ok(FileType(abi::S_IFDIR)),
            abi::DT_BLK => Ok(FileType(abi::S_IFBLK)),
            _ => lstat(&self.path()).map(|m| m.file_type()),
        }
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            // generic
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            // system-specific
            custom_flags: 0,
        }
    }

    pub fn read(&mut self, read: bool) {
        self.read = read;
    }
    pub fn write(&mut self, write: bool) {
        self.write = write;
    }
    pub fn append(&mut self, append: bool) {
        self.append = append;
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

    pub fn custom_flags(&mut self, flags: i32) {
        self.custom_flags = flags;
    }
    pub fn mode(&mut self, _mode: u32) {}

    fn get_access_mode(&self) -> io::Result<c_int> {
        match (self.read, self.write, self.append) {
            (true, false, false) => Ok(abi::O_RDONLY),
            (false, true, false) => Ok(abi::O_WRONLY),
            (true, true, false) => Ok(abi::O_RDWR),
            (false, _, true) => Ok(abi::O_WRONLY | abi::O_APPEND),
            (true, _, true) => Ok(abi::O_RDWR | abi::O_APPEND),
            (false, false, false) => Err(io::Error::from_raw_os_error(libc::EINVAL)),
        }
    }

    fn get_creation_mode(&self) -> io::Result<c_int> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(io::Error::from_raw_os_error(libc::EINVAL));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(io::Error::from_raw_os_error(libc::EINVAL));
                }
            }
        }

        Ok(match (self.create, self.truncate, self.create_new) {
            (false, false, false) => 0,
            (true, false, false) => abi::O_CREAT,
            (false, true, false) => abi::O_TRUNC,
            (true, true, false) => abi::O_CREAT | abi::O_TRUNC,
            (_, _, true) => abi::O_CREAT | abi::O_EXCL,
        })
    }
}

fn cstr(path: &Path) -> io::Result<CString> {
    Ok(CString::new(path.as_os_str().as_bytes())?)
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let flags = opts.get_access_mode()?
            | opts.get_creation_mode()?
            | (opts.custom_flags as c_int & !abi::O_ACCMODE);
        unsafe {
            let mut fd = MaybeUninit::uninit();
            error::SolidError::err_if_negative(abi::SOLID_FS_Open(
                fd.as_mut_ptr(),
                cstr(path)?.as_ptr(),
                flags,
            ))
            .map_err(|e| e.as_io_error())?;
            Ok(File { fd: FileDesc::new(fd.assume_init()) })
        }
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        unsupported()
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.flush()
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.flush()
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        unsupported()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        unsafe {
            let mut out_num_bytes = MaybeUninit::uninit();
            error::SolidError::err_if_negative(abi::SOLID_FS_Read(
                self.fd.raw(),
                buf.as_mut_ptr(),
                buf.len(),
                out_num_bytes.as_mut_ptr(),
            ))
            .map_err(|e| e.as_io_error())?;
            Ok(out_num_bytes.assume_init())
        }
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        unsafe {
            let mut out_num_bytes = MaybeUninit::uninit();
            error::SolidError::err_if_negative(abi::SOLID_FS_Write(
                self.fd.raw(),
                buf.as_ptr(),
                buf.len(),
                out_num_bytes.as_mut_ptr(),
            ))
            .map_err(|e| e.as_io_error())?;
            Ok(out_num_bytes.assume_init())
        }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        error::SolidError::err_if_negative(unsafe { abi::SOLID_FS_Sync(self.fd.raw()) })
            .map_err(|e| e.as_io_error())?;
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            // Casting to `i64` is fine, too large values will end up as
            // negative which will cause an error in `SOLID_FS_Lseek`.
            SeekFrom::Start(off) => (abi::SEEK_SET, off as i64),
            SeekFrom::End(off) => (abi::SEEK_END, off),
            SeekFrom::Current(off) => (abi::SEEK_CUR, off),
        };
        error::SolidError::err_if_negative(unsafe {
            abi::SOLID_FS_Lseek(self.fd.raw(), pos, whence)
        })
        .map_err(|e| e.as_io_error())?;

        // Get the new offset
        unsafe {
            let mut out_offset = MaybeUninit::uninit();
            error::SolidError::err_if_negative(abi::SOLID_FS_Ftell(
                self.fd.raw(),
                out_offset.as_mut_ptr(),
            ))
            .map_err(|e| e.as_io_error())?;
            Ok(out_offset.assume_init() as u64)
        }
    }

    pub fn duplicate(&self) -> io::Result<File> {
        unsupported()
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        unsupported()
    }
}

impl Drop for File {
    fn drop(&mut self) {
        unsafe { abi::SOLID_FS_Close(self.fd.raw()) };
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        error::SolidError::err_if_negative(unsafe { abi::SOLID_FS_Mkdir(cstr(p)?.as_ptr()) })
            .map_err(|e| e.as_io_error())?;
        Ok(())
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File").field("fd", &self.fd.raw()).finish()
    }
}

pub fn unlink(p: &Path) -> io::Result<()> {
    if stat(p)?.file_type().is_dir() {
        Err(io::Error::new_const(io::ErrorKind::IsADirectory, &"is a directory"))
    } else {
        error::SolidError::err_if_negative(unsafe { abi::SOLID_FS_Unlink(cstr(p)?.as_ptr()) })
            .map_err(|e| e.as_io_error())?;
        Ok(())
    }
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    error::SolidError::err_if_negative(unsafe {
        abi::SOLID_FS_Rename(cstr(old)?.as_ptr(), cstr(new)?.as_ptr())
    })
    .map_err(|e| e.as_io_error())?;
    Ok(())
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    error::SolidError::err_if_negative(unsafe {
        abi::SOLID_FS_Chmod(cstr(p)?.as_ptr(), perm.0.into())
    })
    .map_err(|e| e.as_io_error())?;
    Ok(())
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    if stat(p)?.file_type().is_dir() {
        error::SolidError::err_if_negative(unsafe { abi::SOLID_FS_Unlink(cstr(p)?.as_ptr()) })
            .map_err(|e| e.as_io_error())?;
        Ok(())
    } else {
        Err(io::Error::new_const(io::ErrorKind::NotADirectory, &"not a directory"))
    }
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    for child in readdir(path)? {
        let child = child?;
        let child_type = child.file_type()?;
        if child_type.is_dir() {
            remove_dir_all(&child.path())?;
        } else {
            unlink(&child.path())?;
        }
    }
    rmdir(path)
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    // This target doesn't support symlinks
    stat(p)?;
    Err(io::Error::new_const(io::ErrorKind::InvalidInput, &"not a symbolic link"))
}

pub fn symlink(_original: &Path, _link: &Path) -> io::Result<()> {
    // This target doesn't support symlinks
    unsupported()
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    // This target doesn't support symlinks
    unsupported()
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    // This target doesn't support symlinks
    lstat(p)
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    unsafe {
        let mut out_stat = MaybeUninit::uninit();
        error::SolidError::err_if_negative(abi::SOLID_FS_Stat(
            cstr(p)?.as_ptr(),
            out_stat.as_mut_ptr(),
        ))
        .map_err(|e| e.as_io_error())?;
        Ok(FileAttr { stat: out_stat.assume_init() })
    }
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    use crate::fs::File;

    let mut reader = File::open(from)?;
    let mut writer = File::create(to)?;

    io::copy(&mut reader, &mut writer)
}
