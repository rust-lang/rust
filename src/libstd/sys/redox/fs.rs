use crate::os::unix::prelude::*;

use crate::ffi::{OsString, OsStr};
use crate::fmt;
use crate::io::{self, Error, SeekFrom, IoSlice, IoSliceMut};
use crate::path::{Path, PathBuf};
use crate::sync::Arc;
use crate::sys::fd::FileDesc;
use crate::sys::time::SystemTime;
use crate::sys::{cvt, syscall};
use crate::sys_common::{AsInner, FromInner};

pub use crate::sys_common::fs::copy;
pub use crate::sys_common::fs::remove_dir_all;

pub struct File(FileDesc);

#[derive(Clone)]
pub struct FileAttr {
    stat: syscall::Stat,
}

pub struct ReadDir {
    data: Vec<u8>,
    i: usize,
    root: Arc<PathBuf>,
}

struct Dir(FileDesc);

unsafe impl Send for Dir {}
unsafe impl Sync for Dir {}

pub struct DirEntry {
    root: Arc<PathBuf>,
    name: Box<[u8]>
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
    mode: u16,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions { mode: u16 }

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileType { mode: u16 }

#[derive(Debug)]
pub struct DirBuilder { mode: u16 }

impl FileAttr {
    pub fn size(&self) -> u64 { self.stat.st_size as u64 }
    pub fn perm(&self) -> FilePermissions {
        FilePermissions { mode: (self.stat.st_mode as u16) & 0o777 }
    }

    pub fn file_type(&self) -> FileType {
        FileType { mode: self.stat.st_mode as u16 }
    }
}

impl FileAttr {
    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(syscall::TimeSpec {
            tv_sec: self.stat.st_mtime as i64,
            tv_nsec: self.stat.st_mtime_nsec as i32,
        }))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(syscall::TimeSpec {
            tv_sec: self.stat.st_atime as i64,
            tv_nsec: self.stat.st_atime_nsec as i32,
        }))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(syscall::TimeSpec {
            tv_sec: self.stat.st_ctime as i64,
            tv_nsec: self.stat.st_ctime_nsec as i32,
        }))
    }
}

impl AsInner<syscall::Stat> for FileAttr {
    fn as_inner(&self) -> &syscall::Stat { &self.stat }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool { self.mode & 0o222 == 0 }
    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.mode &= !0o222;
        } else {
            self.mode |= 0o222;
        }
    }
    pub fn mode(&self) -> u32 { self.mode as u32 }
}

impl FileType {
    pub fn is_dir(&self) -> bool { self.is(syscall::MODE_DIR) }
    pub fn is_file(&self) -> bool { self.is(syscall::MODE_FILE) }
    pub fn is_symlink(&self) -> bool { self.is(syscall::MODE_SYMLINK) }

    pub fn is(&self, mode: u16) -> bool {
        self.mode & syscall::MODE_TYPE == mode
    }
}

impl FromInner<u32> for FilePermissions {
    fn from_inner(mode: u32) -> FilePermissions {
        FilePermissions { mode: mode as u16 }
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This will only be called from std::fs::ReadDir, which will add a "ReadDir()" frame.
        // Thus the result will be e g 'ReadDir("/home")'
        fmt::Debug::fmt(&*self.root, f)
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        loop {
            let start = self.i;
            let mut i = self.i;
            while i < self.data.len() {
                self.i += 1;
                if self.data[i] == b'\n' {
                    break;
                }
                i += 1;
            }
            if start < self.i {
                let ret = DirEntry {
                    name: self.data[start .. i].to_owned().into_boxed_slice(),
                    root: self.root.clone()
                };
                if ret.name_bytes() != b"." && ret.name_bytes() != b".." {
                    return Some(Ok(ret))
                }
            } else {
                return None;
            }
        }
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.root.join(OsStr::from_bytes(self.name_bytes()))
    }

    pub fn file_name(&self) -> OsString {
        OsStr::from_bytes(self.name_bytes()).to_os_string()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        lstat(&self.path())
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        lstat(&self.path()).map(|m| m.file_type())
    }

    fn name_bytes(&self) -> &[u8] {
        &*self.name
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
            mode: 0o666,
        }
    }

    pub fn read(&mut self, read: bool) { self.read = read; }
    pub fn write(&mut self, write: bool) { self.write = write; }
    pub fn append(&mut self, append: bool) { self.append = append; }
    pub fn truncate(&mut self, truncate: bool) { self.truncate = truncate; }
    pub fn create(&mut self, create: bool) { self.create = create; }
    pub fn create_new(&mut self, create_new: bool) { self.create_new = create_new; }

    pub fn custom_flags(&mut self, flags: i32) { self.custom_flags = flags; }
    pub fn mode(&mut self, mode: u32) { self.mode = mode as u16; }

    fn get_access_mode(&self) -> io::Result<usize> {
        match (self.read, self.write, self.append) {
            (true,  false, false) => Ok(syscall::O_RDONLY),
            (false, true,  false) => Ok(syscall::O_WRONLY),
            (true,  true,  false) => Ok(syscall::O_RDWR),
            (false, _,     true)  => Ok(syscall::O_WRONLY | syscall::O_APPEND),
            (true,  _,     true)  => Ok(syscall::O_RDWR | syscall::O_APPEND),
            (false, false, false) => Err(Error::from_raw_os_error(syscall::EINVAL)),
        }
    }

    fn get_creation_mode(&self) -> io::Result<usize> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) =>
                if self.truncate || self.create || self.create_new {
                    return Err(Error::from_raw_os_error(syscall::EINVAL));
                },
            (_, true) =>
                if self.truncate && !self.create_new {
                    return Err(Error::from_raw_os_error(syscall::EINVAL));
                },
        }

        Ok(match (self.create, self.truncate, self.create_new) {
                (false, false, false) => 0,
                (true,  false, false) => syscall::O_CREAT,
                (false, true,  false) => syscall::O_TRUNC,
                (true,  true,  false) => syscall::O_CREAT | syscall::O_TRUNC,
                (_,      _,    true)  => syscall::O_CREAT | syscall::O_EXCL,
           })
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let flags = syscall::O_CLOEXEC |
                    opts.get_access_mode()? as usize |
                    opts.get_creation_mode()? as usize |
                    (opts.custom_flags as usize & !syscall::O_ACCMODE);
        let fd = cvt(syscall::open(path.to_str().unwrap(), flags | opts.mode as usize))?;
        Ok(File(FileDesc::new(fd)))
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let mut stat = syscall::Stat::default();
        cvt(syscall::fstat(self.0.raw(), &mut stat))?;
        Ok(FileAttr { stat })
    }

    pub fn fsync(&self) -> io::Result<()> {
        cvt(syscall::fsync(self.0.raw()))?;
        Ok(())
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fsync()
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        cvt(syscall::ftruncate(self.0.raw(), size as usize))?;
        Ok(())
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    pub fn flush(&self) -> io::Result<()> { Ok(()) }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            // Casting to `i64` is fine, too large values will end up as
            // negative which will cause an error in `lseek64`.
            SeekFrom::Start(off) => (syscall::SEEK_SET, off as i64),
            SeekFrom::End(off) => (syscall::SEEK_END, off),
            SeekFrom::Current(off) => (syscall::SEEK_CUR, off),
        };
        let n = cvt(syscall::lseek(self.0.raw(), pos as isize, whence))?;
        Ok(n as u64)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        self.0.duplicate().map(File)
    }

    pub fn dup(&self, buf: &[u8]) -> io::Result<File> {
        let fd = cvt(syscall::dup(*self.fd().as_inner() as usize, buf))?;
        Ok(File(FileDesc::new(fd)))
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        set_perm(&self.path()?, perm)
    }

    pub fn path(&self) -> io::Result<PathBuf> {
        let mut buf: [u8; 4096] = [0; 4096];
        let count = cvt(syscall::fpath(*self.fd().as_inner() as usize, &mut buf))?;
        Ok(PathBuf::from(unsafe { String::from_utf8_unchecked(Vec::from(&buf[..count])) }))
    }

    pub fn fd(&self) -> &FileDesc { &self.0 }

    pub fn into_fd(self) -> FileDesc { self.0 }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder { mode: 0o777 }
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        let flags = syscall::O_CREAT | syscall::O_CLOEXEC | syscall::O_DIRECTORY | syscall::O_EXCL;
        let fd = cvt(syscall::open(p.to_str().unwrap(), flags | (self.mode as usize & 0o777)))?;
        let _ = syscall::close(fd);
        Ok(())
    }

    pub fn set_mode(&mut self, mode: u32) {
        self.mode = mode as u16;
    }
}

impl FromInner<usize> for File {
    fn from_inner(fd: usize) -> File {
        File(FileDesc::new(fd))
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b = f.debug_struct("File");
        b.field("fd", &self.0.raw());
        if let Ok(path) = self.path() {
            b.field("path", &path);
        }
        /*
        if let Some((read, write)) = get_mode(fd) {
            b.field("read", &read).field("write", &write);
        }
        */
        b.finish()
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let root = Arc::new(p.to_path_buf());

    let flags = syscall::O_CLOEXEC | syscall::O_RDONLY | syscall::O_DIRECTORY;
    let fd = cvt(syscall::open(p.to_str().unwrap(), flags))?;
    let file = FileDesc::new(fd);
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    Ok(ReadDir { data: data, i: 0, root: root })
}

pub fn unlink(p: &Path) -> io::Result<()> {
    cvt(syscall::unlink(p.to_str().unwrap()))?;
    Ok(())
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let fd = cvt(syscall::open(old.to_str().unwrap(),
                               syscall::O_CLOEXEC | syscall::O_STAT | syscall::O_NOFOLLOW))?;
    let res = cvt(syscall::frename(fd, new.to_str().unwrap()));
    cvt(syscall::close(fd))?;
    res?;
    Ok(())
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    cvt(syscall::chmod(p.to_str().unwrap(), perm.mode as usize))?;
    Ok(())
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    cvt(syscall::rmdir(p.to_str().unwrap()))?;
    Ok(())
}

pub fn readlink(p: &Path) -> io::Result<PathBuf> {
    let fd = cvt(syscall::open(p.to_str().unwrap(),
                               syscall::O_CLOEXEC | syscall::O_SYMLINK | syscall::O_RDONLY))?;
    let mut buf: [u8; 4096] = [0; 4096];
    let res = cvt(syscall::read(fd, &mut buf));
    cvt(syscall::close(fd))?;
    let count = res?;
    Ok(PathBuf::from(unsafe { String::from_utf8_unchecked(Vec::from(&buf[..count])) }))
}

pub fn symlink(src: &Path, dst: &Path) -> io::Result<()> {
    let fd = cvt(syscall::open(dst.to_str().unwrap(),
                               syscall::O_CLOEXEC | syscall::O_SYMLINK |
                               syscall::O_CREAT | syscall::O_WRONLY | 0o777))?;
    let res = cvt(syscall::write(fd, src.to_str().unwrap().as_bytes()));
    cvt(syscall::close(fd))?;
    res?;
    Ok(())
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    Err(Error::from_raw_os_error(syscall::ENOSYS))
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let fd = cvt(syscall::open(p.to_str().unwrap(), syscall::O_CLOEXEC | syscall::O_STAT))?;
    let file = File(FileDesc::new(fd));
    file.file_attr()
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    let fd = cvt(syscall::open(p.to_str().unwrap(),
                               syscall::O_CLOEXEC | syscall::O_STAT | syscall::O_NOFOLLOW))?;
    let file = File(FileDesc::new(fd));
    file.file_attr()
}

pub fn canonicalize(p: &Path) -> io::Result<PathBuf> {
    let fd = cvt(syscall::open(p.to_str().unwrap(), syscall::O_CLOEXEC | syscall::O_STAT))?;
    let file = File(FileDesc::new(fd));
    file.path()
}
