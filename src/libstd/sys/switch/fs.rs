use crate::ffi::{OsString, CStr, OsStr};
use crate::os::switch::ffi::OsStrExt;
use crate::fmt;
use crate::hash::Hash;
use crate::io::{self, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sys::time::{SystemTime, UNIX_EPOCH};
use crate::sys::unsupported;
use crate::sync::atomic::{AtomicU64, Ordering};

use nnsdk::fs::{FileHandle, DirectoryEntry as NinDirEntry};
use nnsdk::fs::DirectoryEntryType_DirectoryEntryType_Directory as NN_ENTRY_DIR;
use nnsdk::fs::DirectoryEntryType_DirectoryEntryType_File as NN_ENTRY_FILE;
use crate::ffi::CString;

macro_rules! r_try {
    ($expr:expr) => {
        {
            let rc = $expr;
            if rc == 0 {
                Ok(())
            } else {
                Err(io::Error::from_raw_os_error(rc as _))
            }
        }
    };
}

#[derive(Debug)]
pub struct FileAttr {
    size: AtomicU64,
    file_type: FileType
}

pub struct ReadDir {
    inner_iter: crate::vec::IntoIter<NinDirEntry>
}

pub struct DirEntry {
    path: PathBuf,
    file_attr: FileAttr,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    read_only: bool
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum FileType {
    Dir,
    File
}

#[derive(Debug)]
pub struct DirBuilder {}

impl Clone for FileAttr {
    fn clone(&self) -> Self {
        Self {
            size: AtomicU64::new(self.size.load(Ordering::SeqCst)),
            file_type: self.file_type
        }
    }
}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.size.load(Ordering::SeqCst)
    }

    pub fn set_size(&self, size: u64) {
        self.size.store(size, Ordering::SeqCst);
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions { read_only: false }
    }

    pub fn file_type(&self) -> FileType {
        self.file_type
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(UNIX_EPOCH)
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(UNIX_EPOCH)
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(UNIX_EPOCH)
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.read_only
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        self.read_only = readonly;
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        if let FileType::Dir = self {
            true
        } else {
            false
        }
    }

    pub fn is_file(&self) -> bool {
        if let FileType::File = self {
            true
        } else {
            false
        }
    }

    pub fn is_symlink(&self) -> bool {
        false
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[Directory]")
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        let val = self.inner_iter.next()?;

        let len = unsafe { libc::strlen(val.name.as_ptr()) };
        let path = Path::new(OsStr::from_bytes(&val.name[..len])).to_owned();

        let file_type = match val.type_ as u32 {
            NN_ENTRY_DIR => FileType::Dir,
            NN_ENTRY_FILE => FileType::File,
            _ => panic!("Invalid entry type in directory")
        };

        let file_attr = FileAttr {
            size: AtomicU64::new(val.fileSize as u64),
            file_type
        };

        Some(Ok(DirEntry {
            path, file_attr
        }))
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn file_name(&self) -> OsString {
        match self.path.file_name() {
            Some(file_name) => file_name.to_os_string(),
            None => panic!("Could not get file name of DirEntry")
        }
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        Ok(self.file_attr.clone())
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(self.file_attr.file_type)
    }
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    flags: u64,
    truncate: bool
}

const READ_MODE: u64 = 1;
const WRITE_MODE: u64 = 2;
const APPEND_MODE: u64 = 4;

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            flags: 0,
            truncate: false
        }
    }

    pub fn read(&mut self, read: bool) {
        if read {
            self.flags |= READ_MODE;
        } else {
            self.flags &= !READ_MODE;
        }
    }
    pub fn write(&mut self, write: bool) {
        if write {
            self.flags |= WRITE_MODE;
        } else {
            self.flags &= !WRITE_MODE;
        }
    }
    pub fn append(&mut self, append: bool) {
        if append {
            self.flags |= APPEND_MODE;
        } else {
            self.flags &= !APPEND_MODE;
        }
    }
    pub fn truncate(&mut self, truncate: bool) {
        self.truncate = truncate;
    }
    pub fn create(&mut self, _create: bool) {
        
    }

    pub fn create_new(&mut self, _create_new: bool) {
        panic!("File create new not supported yet")
    }
}

pub struct File {
    inner: FileHandle,
    pos: AtomicU64,
    attr: FileAttr
}

impl crate::ops::Drop for File {
    fn drop(&mut self) {
        if self.inner.handle.is_null() {
            return;
        }
        unsafe {
            nnsdk::fs::CloseFile(
                self.inner
            );
        }
    }
}

fn cstr(path: &Path) -> io::Result<CString> {
    CString::new(
        path.to_str()
            .ok_or(io::Error::from(io::ErrorKind::InvalidInput))?
            .as_bytes()
    ).map_err(io::Error::from)
}

macro_rules! ret_if_null {
    ($expr:expr) => {
        if ($expr.handle).is_null() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot treat directory as file"
            ))
        }
    };
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path = cstr(path)?;

        let mut inner = FileHandle { handle: 0 as _ };
        
        let mut entry_type = 0u32;

        unsafe {
            r_try!(
                nnsdk::fs::GetEntryType(
                    &mut entry_type,
                    path.as_ptr() as _
                )
            )?;
        }
        
        let file_type = match entry_type {
            NN_ENTRY_DIR => FileType::Dir,
            NN_ENTRY_FILE => FileType::File,
            _ => panic!("Invalid entry type in directory")
        };

        if let FileType::Dir = file_type {
            let attr = FileAttr {
                size: AtomicU64::new(0),
                file_type
            };
            return Ok(File {
                inner, pos: AtomicU64::new(0), attr
            })
        }

        unsafe { 
            r_try!(
                nnsdk::fs::OpenFile(
                    &mut inner,
                    path.as_ptr() as _,
                    opts.flags as _
                )
            )?;
        }

        if inner.handle.is_null() {
            Err(io::Error::new(io::ErrorKind::NotFound, "Returned file handle was null"))
        } else {
            let mut size = 0;
             unsafe { 
                r_try!(nnsdk::fs::GetFileSize(&mut size, inner))?;
            }
            
            let pos = if opts.flags & APPEND_MODE != 0 {
                AtomicU64::new(size as u64)
            } else {
                AtomicU64::new(0)
            };

            let attr = stat_internal(&path, size as _)?;

            let file = File { inner, pos, attr };

            if opts.truncate {
                file.truncate(0)?;
            }

            Ok(file)
        }
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        Ok(self.attr.clone())
    }

    pub fn fsync(&self) -> io::Result<()> {
        ret_if_null!(self.inner);
        unsafe { r_try!(nnsdk::fs::FlushFile(self.inner)) }
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fsync()
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        ret_if_null!(self.inner);
        let rc = unsafe {
            nnsdk::fs::SetFileSize(self.inner, size as _)
        };

        self.attr.set_size(size);

        r_try!(rc)
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        ret_if_null!(self.inner);
        let mut out_size = 0;
        let rc = unsafe {
            nnsdk::fs::ReadFile1(
                &mut out_size,
                self.inner,
                self.pos() as _,
                buf.as_ptr() as _,
                buf.len() as _
            )
        };

        if rc == 0 {
            self.pos.fetch_add(out_size, Ordering::SeqCst);
            Ok(out_size as usize)
        } else {
            Err(io::Error::from_raw_os_error(rc as _))
        }
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let mut read = 0;
        for mut buf in bufs {
            let amt = self.read(&mut buf)?;
            read += amt;
            if amt != buf.len() {
                break
            }
        }
        Ok(read)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        ret_if_null!(self.inner);
        let rc = unsafe {
            nnsdk::fs::WriteFile(
                self.inner,
                self.pos() as _,
                buf.as_ptr() as _,
                buf.len() as u64,
                &nnsdk::fs::WriteOption { flags: 0 }
            )
        };

        if rc == 0 {
            self.pos.fetch_add(buf.len() as u64, Ordering::SeqCst);
            if self.pos() > self.attr.size() {
                self.attr.set_size(self.pos());
            }
            Ok(buf.len())
        } else {
            Err(io::Error::from_raw_os_error(rc as _))
        }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let mut written = 0;
        for buf in bufs {
            let amt = self.write(&buf)?;
            written += amt;
            if amt != buf.len() {
                break
            }
        }
        Ok(written)
    }

    pub fn flush(&self) -> io::Result<()> {
        self.fsync()
    }

    fn pos(&self) -> u64 {
        self.pos.load(Ordering::SeqCst)
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Start(offset) => {
                self.pos.store(offset, Ordering::SeqCst);
            },
            SeekFrom::Current(offset) => {
                let pos = (self.pos.load(Ordering::SeqCst) as i64) + offset;
                if pos < 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Attempted to seek to an invalid or negative offset"
                    ))
                }
                self.pos.store(pos as u64, Ordering::SeqCst);
            },
            SeekFrom::End(offset) => {
                if offset > 0 || (-offset as u64) > self.attr.size() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Attempted to seek to an invalid or negative offset"
                    ))
                }
                self.pos.store(self.attr.size() + (-offset as u64), Ordering::SeqCst);
            },
        };

        Ok(self.pos.load(Ordering::SeqCst))
    }

    pub fn duplicate(&self) -> io::Result<File> {
        // This feels super wrong and will probably break something
        Ok(File {
            inner: self.inner.clone(),
            pos: AtomicU64::new(self.pos()),
            attr: self.attr.clone()
        })
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        Ok(())
    }

    pub fn diverge(&self) -> ! {
        panic!("file diverge")
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, path: &Path) -> io::Result<()> {
        let path = cstr(path)?;

        unsafe {
            r_try!(nnsdk::fs::CreateDirectory(path.as_ptr() as *const _))
        }
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[Open File]")
    }
}

pub fn readdir(path: &Path) -> io::Result<ReadDir> {
    let path = cstr(path)?;

    let mut dir_handle = nnsdk::fs::DirectoryHandle { handle: 0 as *mut _ };
    let mut count: i64 = 0;

    unsafe {
        r_try!(
            nnsdk::fs::OpenDirectory(
                &mut dir_handle as _,
                path.as_ptr() as _,
                nnsdk::fs::OpenDirectoryMode_OpenDirectoryMode_All as _
            )
        )?;

        r_try!(
            nnsdk::fs::GetDirectoryEntryCount(
                &mut count as _,
                dir_handle
            )
        )?;
    }

    let mut entries: Vec<NinDirEntry> = Vec::with_capacity(count as usize);

    unsafe {
        entries.set_len(count as usize);

        r_try!(
            nnsdk::fs::ReadDirectory(
                &mut count,
                entries.as_mut_ptr(),
                dir_handle,
                count
            )
        )?;

        nnsdk::fs::CloseDirectory(dir_handle);
    }

    Ok(ReadDir {
        inner_iter: entries.into_iter()
    })
}

pub fn unlink(_p: &Path) -> io::Result<()> {
    unsupported()
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let stat = stat(old)?;
    let old = cstr(old)?;
    let new = cstr(new)?;

    r_try!(if let FileType::File = stat.file_type() {
        unsafe {
            nnsdk::fs::RenameFile(old.as_ptr() as _, new.as_ptr() as _)
        }
    } else {
        unsafe {
            nnsdk::fs::RenameDirectory(old.as_ptr() as _, new.as_ptr() as _)
        }
    })
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    Ok(())
}

pub fn rmdir(path: &Path) -> io::Result<()> {
    let path = cstr(path)?;

    if (nnsdk::fs::DeleteDirectory as *const ()).is_null() {
        panic!("DeleteDirectory is null");
    }

    unsafe {
        r_try!(nnsdk::fs::DeleteDirectory(path.as_ptr() as _))
    }
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let path = cstr(path)?;

    if (nnsdk::fs::DeleteDirectoryRecursively as *const ()).is_null() {
        panic!("DeleteDirectoryRecursively is null");
    }

    unsafe {
        r_try!(nnsdk::fs::DeleteDirectoryRecursively(path.as_ptr() as _))
    }
}

pub fn readlink(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn symlink(_src: &Path, _dst: &Path) -> io::Result<()> {
    unsupported()
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    unsupported()
}

fn get_entry_type(cstr: &CStr) -> io::Result<FileType> {
    let mut entry_type: u32 = 0;
    unsafe {
        r_try!(
            nnsdk::fs::GetEntryType(
                &mut entry_type,
                cstr.as_ptr() as _
            )
        )?;
    }

    match entry_type {
        NN_ENTRY_DIR => Ok(FileType::Dir),
        NN_ENTRY_FILE => Ok(FileType::File),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Could not stat directory type"
        ))
    }
}

fn stat_internal(cstr: &CStr, size: u64) -> io::Result<FileAttr> {
    let file_type = get_entry_type(cstr)?;

    Ok(FileAttr {
        size: AtomicU64::new(size),
        file_type
    })
}

pub fn stat(path: &Path) -> io::Result<FileAttr> {
    match get_entry_type(&(cstr(path)?))? {
        file_type @ FileType::Dir => Ok(FileAttr { size: AtomicU64::new(0), file_type }),
        file_type @ FileType::File => Ok(FileAttr { size: AtomicU64::new(0), file_type }),
        //_ => panic!("Bad entry type")
    }
    //File::open(path, &OpenOptions::new())?.file_attr()
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    stat(path)
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    crate::io::copy(
        &mut crate::fs::File::open(from)?,
        &mut crate::fs::File::create(to)?,
    )
}
