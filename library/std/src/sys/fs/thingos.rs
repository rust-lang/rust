//! ThingOS filesystem implementation.
//!
//! All I/O goes through the raw syscall interface defined in
//! `sys/pal/thingos/common.rs`.

use crate::ffi::OsString;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::os::fd::AsRawFd;
use crate::path::{Path, PathBuf};
use crate::sys::fd::FileDesc;
pub use crate::sys::fs::common::{Dir, exists};
use crate::sys::pal::common::{
    DT_DIR, DT_LNK, DT_REG, O_APPEND, O_CLOEXEC, O_CREAT, O_EXCL, O_RDONLY, O_RDWR, O_TRUNC,
    O_WRONLY, SEEK_CUR, SEEK_END, SEEK_SET, SYS_CHMOD, SYS_FCHMOD, SYS_FSTAT, SYS_FSYNC,
    SYS_FTRUNCATE, SYS_GETDENTS, SYS_LSTAT, SYS_MKDIR, SYS_OPEN, SYS_READLINK, SYS_RENAME,
    SYS_RMDIR, SYS_STAT, SYS_SYMLINK, SYS_UNLINK, Stat, cvt, raw_syscall6,
};
use crate::sys::time::SystemTime;
use crate::sys::{AsInner, AsInnerMut, FromInner, IntoInner};

// ── FileType ─────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileType(u8);

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.0 == DT_DIR
    }
    pub fn is_file(&self) -> bool {
        self.0 == DT_REG
    }
    pub fn is_symlink(&self) -> bool {
        self.0 == DT_LNK
    }
}

// ── FilePermissions ───────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FilePermissions {
    mode: u8,
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.mode & 0b10 == 0
    }
    pub fn set_readonly(&mut self, ro: bool) {
        if ro {
            self.mode &= !0b10;
        } else {
            self.mode |= 0b10;
        }
    }
}

// ── FileTimes ─────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    pub modified: u64,
    pub accessed: u64,
}

impl FileTimes {
    pub fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = t.as_nanos_since_epoch() as u64;
    }
    pub fn set_modified(&mut self, t: SystemTime) {
        self.modified = t.as_nanos_since_epoch() as u64;
    }
}

// ── FileAttr ─────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
pub struct FileAttr {
    stat: Stat,
}

impl FileAttr {
    fn from_stat(stat: Stat) -> FileAttr {
        FileAttr { stat }
    }

    pub fn size(&self) -> u64 {
        self.stat.size
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions { mode: self.stat.perm }
    }

    pub fn file_type(&self) -> FileType {
        FileType(self.stat.file_type)
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        if self.stat.modified == 0 {
            Err(io::Error::other("modified time not available"))
        } else {
            Ok(SystemTime::from_nanos_since_epoch(self.stat.modified as u128))
        }
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        if self.stat.accessed == 0 {
            Err(io::Error::other("accessed time not available"))
        } else {
            Ok(SystemTime::from_nanos_since_epoch(self.stat.accessed as u128))
        }
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        if self.stat.created == 0 {
            Err(io::Error::other("creation time not available"))
        } else {
            Ok(SystemTime::from_nanos_since_epoch(self.stat.created as u128))
        }
    }
}

// ── OpenOptions ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
        }
    }

    pub fn read(&mut self, v: bool) {
        self.read = v;
    }
    pub fn write(&mut self, v: bool) {
        self.write = v;
    }
    pub fn append(&mut self, v: bool) {
        self.append = v;
    }
    pub fn truncate(&mut self, v: bool) {
        self.truncate = v;
    }
    pub fn create(&mut self, v: bool) {
        self.create = v;
    }
    pub fn create_new(&mut self, v: bool) {
        self.create_new = v;
    }
    pub fn custom_flags(&mut self, _flags: i32) {}
    pub fn mode(&mut self, _mode: u32) {}

    fn flags(&self) -> u64 {
        let mut flags = O_CLOEXEC;
        if self.read && self.write {
            flags |= O_RDWR;
        } else if self.write {
            flags |= O_WRONLY;
        } else {
            flags |= O_RDONLY;
        }
        if self.append {
            flags |= O_APPEND;
        }
        if self.truncate {
            flags |= O_TRUNC;
        }
        if self.create_new {
            flags |= O_CREAT | O_EXCL;
        } else if self.create {
            flags |= O_CREAT;
        }
        flags
    }
}

// ── File ─────────────────────────────────────────────────────────────────────

pub struct File(FileDesc);

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path_bytes = path_to_bytes(path)?;
        let flags = opts.flags();
        let fd = cvt(unsafe {
            raw_syscall6(
                SYS_OPEN,
                path_bytes.as_ptr() as u64,
                path_bytes.len() as u64,
                flags,
                0o666,
                0,
                0,
            )
        })?;
        // SAFETY: kernel returned a valid, owned fd.
        unsafe {
            use crate::os::fd::{FromRawFd, OwnedFd};
            Ok(File(FileDesc::from_inner(OwnedFd::from_raw_fd(fd as i32))))
        }
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let mut stat = Stat::default();
        cvt(unsafe {
            raw_syscall6(
                SYS_FSTAT,
                self.0.as_inner().as_raw_fd() as u64,
                &raw mut stat as u64,
                0,
                0,
                0,
                0,
            )
        })?;
        Ok(FileAttr::from_stat(stat))
    }

    pub fn fsync(&self) -> io::Result<()> {
        cvt(unsafe {
            raw_syscall6(SYS_FSYNC, self.0.as_inner().as_raw_fd() as u64, 0, 0, 0, 0, 0)
        })?;
        Ok(())
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fsync()
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        cvt(unsafe {
            raw_syscall6(
                SYS_FTRUNCATE,
                self.0.as_inner().as_raw_fd() as u64,
                size,
                0,
                0,
                0,
                0,
            )
        })?;
        Ok(())
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.0.read_buf(cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, offset) = match pos {
            SeekFrom::Start(n) => (SEEK_SET, n as i64),
            SeekFrom::End(n) => (SEEK_END, n),
            SeekFrom::Current(n) => (SEEK_CUR, n),
        };
        let ret = cvt(unsafe {
            raw_syscall6(
                crate::sys::pal::common::SYS_LSEEK,
                self.0.as_inner().as_raw_fd() as u64,
                offset as u64,
                whence,
                0,
                0,
                0,
            )
        })?;
        Ok(ret as u64)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        self.0.duplicate().map(File)
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        cvt(unsafe {
            raw_syscall6(
                SYS_FCHMOD,
                self.0.as_inner().as_raw_fd() as u64,
                perm.mode as u64,
                0,
                0,
                0,
                0,
            )
        })?;
        Ok(())
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        // TODO: wire up once SYS_UTIMES is defined.
        crate::sys::pal::unsupported()
    }
}

impl AsInner<FileDesc> for File {
    fn as_inner(&self) -> &FileDesc {
        &self.0
    }
}

impl AsInnerMut<FileDesc> for File {
    fn as_inner_mut(&mut self) -> &mut FileDesc {
        &mut self.0
    }
}

impl IntoInner<FileDesc> for File {
    fn into_inner(self) -> FileDesc {
        self.0
    }
}

impl FromInner<FileDesc> for File {
    fn from_inner(fd: FileDesc) -> File {
        File(fd)
    }
}

impl crate::fmt::Debug for File {
    fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
        write!(f, "File(fd={})", self.0.as_inner().as_raw_fd())
    }
}

// ── Directory operations ──────────────────────────────────────────────────────

pub fn stat(path: &Path) -> io::Result<FileAttr> {
    let path_bytes = path_to_bytes(path)?;
    let mut s = Stat::default();
    cvt(unsafe {
        raw_syscall6(SYS_STAT, path_bytes.as_ptr() as u64, path_bytes.len() as u64, &raw mut s as u64, 0, 0, 0)
    })?;
    Ok(FileAttr::from_stat(s))
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    let path_bytes = path_to_bytes(path)?;
    let mut s = Stat::default();
    cvt(unsafe {
        raw_syscall6(SYS_LSTAT, path_bytes.as_ptr() as u64, path_bytes.len() as u64, &raw mut s as u64, 0, 0, 0)
    })?;
    Ok(FileAttr::from_stat(s))
}

pub fn unlink(path: &Path) -> io::Result<()> {
    let b = path_to_bytes(path)?;
    cvt(unsafe { raw_syscall6(SYS_UNLINK, b.as_ptr() as u64, b.len() as u64, 0, 0, 0, 0) })?;
    Ok(())
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let ob = path_to_bytes(old)?;
    let nb = path_to_bytes(new)?;
    cvt(unsafe {
        raw_syscall6(
            SYS_RENAME,
            ob.as_ptr() as u64,
            ob.len() as u64,
            nb.as_ptr() as u64,
            nb.len() as u64,
            0,
            0,
        )
    })?;
    Ok(())
}

pub fn rmdir(path: &Path) -> io::Result<()> {
    let b = path_to_bytes(path)?;
    cvt(unsafe { raw_syscall6(SYS_RMDIR, b.as_ptr() as u64, b.len() as u64, 0, 0, 0, 0) })?;
    Ok(())
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    // Walk the directory recursively.
    for entry in readdir(path)? {
        let entry = entry?;
        let child = entry.path();
        if entry.file_type()?.is_dir() {
            remove_dir_all(&child)?;
        } else {
            unlink(&child)?;
        }
    }
    rmdir(path)
}

pub fn set_perm(path: &Path, perm: FilePermissions) -> io::Result<()> {
    let b = path_to_bytes(path)?;
    cvt(unsafe {
        raw_syscall6(SYS_CHMOD, b.as_ptr() as u64, b.len() as u64, perm.mode as u64, 0, 0, 0)
    })?;
    Ok(())
}

pub fn set_times(_path: &Path, _times: FileTimes) -> io::Result<()> {
    crate::sys::pal::unsupported()
}

pub fn set_times_nofollow(_path: &Path, _times: FileTimes) -> io::Result<()> {
    crate::sys::pal::unsupported()
}

pub fn readlink(path: &Path) -> io::Result<PathBuf> {
    let b = path_to_bytes(path)?;
    let mut buf = vec![0u8; 4096];
    let len = cvt(unsafe {
        raw_syscall6(
            SYS_READLINK,
            b.as_ptr() as u64,
            b.len() as u64,
            buf.as_mut_ptr() as u64,
            buf.len() as u64,
            0,
            0,
        )
    })? as usize;
    let s = core::str::from_utf8(&buf[..len]).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "readlink returned non-UTF-8")
    })?;
    Ok(PathBuf::from(s))
}

pub fn symlink(original: &Path, link: &Path) -> io::Result<()> {
    let ob = path_to_bytes(original)?;
    let lb = path_to_bytes(link)?;
    cvt(unsafe {
        raw_syscall6(
            SYS_SYMLINK,
            ob.as_ptr() as u64,
            ob.len() as u64,
            lb.as_ptr() as u64,
            lb.len() as u64,
            0,
            0,
        )
    })?;
    Ok(())
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    crate::sys::pal::unsupported()
}

pub fn canonicalize(path: &Path) -> io::Result<PathBuf> {
    // TODO: Implement a real path canonicalization (resolve symlinks, remove ../).
    // Currently returns the path unchanged as a placeholder. This is incorrect
    // for paths containing symlinks or relative components, but is safe to use
    // for simple absolute paths until a SYS_REALPATH syscall is available.
    Ok(path.to_path_buf())
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    use crate::io::Read;
    let mut reader_opts = OpenOptions::new();
    reader_opts.read(true);
    let mut writer_opts = OpenOptions::new();
    writer_opts.write(true);
    writer_opts.create(true);
    writer_opts.truncate(true);
    let mut reader = File::open(from, &reader_opts)?;
    let writer = File::open(to, &writer_opts)?;
    let mut buf = vec![0u8; 65536];
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write(&buf[..n])?;
        total += n as u64;
    }
    Ok(total)
}

// ── DirBuilder ────────────────────────────────────────────────────────────────

pub struct DirBuilder {}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, path: &Path) -> io::Result<()> {
        let b = path_to_bytes(path)?;
        cvt(unsafe {
            raw_syscall6(SYS_MKDIR, b.as_ptr() as u64, b.len() as u64, 0o755, 0, 0, 0)
        })?;
        Ok(())
    }
}

// ── ReadDir / DirEntry ────────────────────────────────────────────────────────

pub struct ReadDir {
    fd: FileDesc,
    root: PathBuf,
    /// Remaining raw bytes from the last `SYS_GETDENTS` call.
    buf: Vec<u8>,
    buf_pos: usize,
    buf_end: usize,
    done: bool,
}

pub struct DirEntry {
    name: OsString,
    file_type: FileType,
    root: PathBuf,
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.root.join(&self.name)
    }

    pub fn file_name(&self) -> OsString {
        self.name.clone()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        stat(&self.path())
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(self.file_type)
    }

    pub fn ino(&self) -> u64 {
        0
    }
}

pub fn readdir(path: &Path) -> io::Result<ReadDir> {
    // Open the directory.
    let mut opts = OpenOptions::new();
    opts.read(true);
    let file = File::open(path, &opts)?;
    Ok(ReadDir {
        fd: file.0,
        root: path.to_path_buf(),
        buf: vec![0u8; 4096],
        buf_pos: 0,
        buf_end: 0,
        done: false,
    })
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        loop {
            if self.done {
                return None;
            }

            // Refill buffer when exhausted.
            if self.buf_pos >= self.buf_end {
                let ret = unsafe {
                    raw_syscall6(
                        SYS_GETDENTS,
                        self.fd.as_inner().as_raw_fd() as u64,
                        self.buf.as_mut_ptr() as u64,
                        self.buf.len() as u64,
                        0,
                        0,
                        0,
                    )
                };
                match cvt(ret) {
                    Err(e) => return Some(Err(e)),
                    Ok(0) => {
                        self.done = true;
                        return None;
                    }
                    Ok(n) => {
                        self.buf_pos = 0;
                        self.buf_end = n as usize;
                    }
                }
            }

            // Parse one dirent from the buffer.
            if self.buf_pos + 19 > self.buf_end {
                self.done = true;
                return None;
            }

            // Dirent layout: ino(8) + off(8) + reclen(2) + type(1) + name(...)
            let base = self.buf_pos;
            let reclen = u16::from_ne_bytes([self.buf[base + 16], self.buf[base + 17]]) as usize;
            let file_type_byte = self.buf[base + 18];
            let name_start = base + 19;
            let name_end_max = base + reclen;

            self.buf_pos += reclen;

            if reclen == 0 || name_end_max > self.buf_end {
                continue;
            }

            // Name is NUL-terminated.
            let name_bytes = &self.buf[name_start..name_end_max];
            let nul = name_bytes.iter().position(|&b| b == 0).unwrap_or(name_bytes.len());
            let name_bytes = &name_bytes[..nul];

            // Skip "." and "..".
            if name_bytes == b"." || name_bytes == b".." {
                continue;
            }

            let name = OsString::from(
                core::str::from_utf8(name_bytes).unwrap_or_default(),
            );

            let file_type = match file_type_byte {
                d if d == DT_DIR => FileType(DT_DIR),
                d if d == DT_LNK => FileType(DT_LNK),
                _ => FileType(DT_REG),
            };

            return Some(Ok(DirEntry { name, file_type, root: self.root.clone() }));
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert a `Path` to a NUL-terminated byte slice.
fn path_to_bytes(path: &Path) -> io::Result<Vec<u8>> {
    let s = path.as_os_str().as_encoded_bytes();
    // Check for embedded NULs.
    if s.contains(&0) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "path contains null byte"));
    }
    let mut v = s.to_vec();
    v.push(0); // NUL-terminate
    Ok(v)
}
