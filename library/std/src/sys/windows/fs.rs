use crate::os::windows::prelude::*;

use crate::ffi::OsString;
use crate::fmt;
use crate::io::{self, Error, IoSlice, IoSliceMut, SeekFrom};
use crate::mem;
use crate::os::windows::io::{AsHandle, BorrowedHandle};
use crate::path::{Path, PathBuf};
use crate::ptr;
use crate::slice;
use crate::sync::Arc;
use crate::sys::handle::Handle;
use crate::sys::time::SystemTime;
use crate::sys::{c, cvt};
use crate::sys_common::{AsInner, FromInner, IntoInner};

use super::path::maybe_verbatim;
use super::to_u16s;

pub struct File {
    handle: Handle,
}

#[derive(Clone)]
pub struct FileAttr {
    attributes: c::DWORD,
    creation_time: c::FILETIME,
    last_access_time: c::FILETIME,
    last_write_time: c::FILETIME,
    file_size: u64,
    reparse_tag: c::DWORD,
    volume_serial_number: Option<u32>,
    number_of_links: Option<u32>,
    file_index: Option<u64>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileType {
    attributes: c::DWORD,
    reparse_tag: c::DWORD,
}

pub struct ReadDir {
    handle: FindNextFileHandle,
    root: Arc<PathBuf>,
    first: Option<c::WIN32_FIND_DATAW>,
}

struct FindNextFileHandle(c::HANDLE);

unsafe impl Send for FindNextFileHandle {}
unsafe impl Sync for FindNextFileHandle {}

pub struct DirEntry {
    root: Arc<PathBuf>,
    data: c::WIN32_FIND_DATAW,
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
    custom_flags: u32,
    access_mode: Option<c::DWORD>,
    attributes: c::DWORD,
    share_mode: c::DWORD,
    security_qos_flags: c::DWORD,
    security_attributes: usize, // FIXME: should be a reference
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    attrs: c::DWORD,
}

#[derive(Debug)]
pub struct DirBuilder;

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This will only be called from std::fs::ReadDir, which will add a "ReadDir()" frame.
        // Thus the result will be e g 'ReadDir("C:\")'
        fmt::Debug::fmt(&*self.root, f)
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;
    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        if let Some(first) = self.first.take() {
            if let Some(e) = DirEntry::new(&self.root, &first) {
                return Some(Ok(e));
            }
        }
        unsafe {
            let mut wfd = mem::zeroed();
            loop {
                if c::FindNextFileW(self.handle.0, &mut wfd) == 0 {
                    if c::GetLastError() == c::ERROR_NO_MORE_FILES {
                        return None;
                    } else {
                        return Some(Err(Error::last_os_error()));
                    }
                }
                if let Some(e) = DirEntry::new(&self.root, &wfd) {
                    return Some(Ok(e));
                }
            }
        }
    }
}

impl Drop for FindNextFileHandle {
    fn drop(&mut self) {
        let r = unsafe { c::FindClose(self.0) };
        debug_assert!(r != 0);
    }
}

impl DirEntry {
    fn new(root: &Arc<PathBuf>, wfd: &c::WIN32_FIND_DATAW) -> Option<DirEntry> {
        match &wfd.cFileName[0..3] {
            // check for '.' and '..'
            &[46, 0, ..] | &[46, 46, 0, ..] => return None,
            _ => {}
        }

        Some(DirEntry { root: root.clone(), data: *wfd })
    }

    pub fn path(&self) -> PathBuf {
        self.root.join(&self.file_name())
    }

    pub fn file_name(&self) -> OsString {
        let filename = super::truncate_utf16_at_nul(&self.data.cFileName);
        OsString::from_wide(filename)
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(FileType::new(
            self.data.dwFileAttributes,
            /* reparse_tag = */ self.data.dwReserved0,
        ))
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        Ok(FileAttr {
            attributes: self.data.dwFileAttributes,
            creation_time: self.data.ftCreationTime,
            last_access_time: self.data.ftLastAccessTime,
            last_write_time: self.data.ftLastWriteTime,
            file_size: ((self.data.nFileSizeHigh as u64) << 32) | (self.data.nFileSizeLow as u64),
            reparse_tag: if self.data.dwFileAttributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                // reserved unless this is a reparse point
                self.data.dwReserved0
            } else {
                0
            },
            volume_serial_number: None,
            number_of_links: None,
            file_index: None,
        })
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
            access_mode: None,
            share_mode: c::FILE_SHARE_READ | c::FILE_SHARE_WRITE | c::FILE_SHARE_DELETE,
            attributes: 0,
            security_qos_flags: 0,
            security_attributes: 0,
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

    pub fn custom_flags(&mut self, flags: u32) {
        self.custom_flags = flags;
    }
    pub fn access_mode(&mut self, access_mode: u32) {
        self.access_mode = Some(access_mode);
    }
    pub fn share_mode(&mut self, share_mode: u32) {
        self.share_mode = share_mode;
    }
    pub fn attributes(&mut self, attrs: u32) {
        self.attributes = attrs;
    }
    pub fn security_qos_flags(&mut self, flags: u32) {
        // We have to set `SECURITY_SQOS_PRESENT` here, because one of the valid flags we can
        // receive is `SECURITY_ANONYMOUS = 0x0`, which we can't check for later on.
        self.security_qos_flags = flags | c::SECURITY_SQOS_PRESENT;
    }
    pub fn security_attributes(&mut self, attrs: c::LPSECURITY_ATTRIBUTES) {
        self.security_attributes = attrs as usize;
    }

    fn get_access_mode(&self) -> io::Result<c::DWORD> {
        const ERROR_INVALID_PARAMETER: i32 = 87;

        match (self.read, self.write, self.append, self.access_mode) {
            (.., Some(mode)) => Ok(mode),
            (true, false, false, None) => Ok(c::GENERIC_READ),
            (false, true, false, None) => Ok(c::GENERIC_WRITE),
            (true, true, false, None) => Ok(c::GENERIC_READ | c::GENERIC_WRITE),
            (false, _, true, None) => Ok(c::FILE_GENERIC_WRITE & !c::FILE_WRITE_DATA),
            (true, _, true, None) => {
                Ok(c::GENERIC_READ | (c::FILE_GENERIC_WRITE & !c::FILE_WRITE_DATA))
            }
            (false, false, false, None) => Err(Error::from_raw_os_error(ERROR_INVALID_PARAMETER)),
        }
    }

    fn get_creation_mode(&self) -> io::Result<c::DWORD> {
        const ERROR_INVALID_PARAMETER: i32 = 87;

        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(Error::from_raw_os_error(ERROR_INVALID_PARAMETER));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(Error::from_raw_os_error(ERROR_INVALID_PARAMETER));
                }
            }
        }

        Ok(match (self.create, self.truncate, self.create_new) {
            (false, false, false) => c::OPEN_EXISTING,
            (true, false, false) => c::OPEN_ALWAYS,
            (false, true, false) => c::TRUNCATE_EXISTING,
            (true, true, false) => c::CREATE_ALWAYS,
            (_, _, true) => c::CREATE_NEW,
        })
    }

    fn get_flags_and_attributes(&self) -> c::DWORD {
        self.custom_flags
            | self.attributes
            | self.security_qos_flags
            | if self.create_new { c::FILE_FLAG_OPEN_REPARSE_POINT } else { 0 }
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path = maybe_verbatim(path)?;
        let handle = unsafe {
            c::CreateFileW(
                path.as_ptr(),
                opts.get_access_mode()?,
                opts.share_mode,
                opts.security_attributes as *mut _,
                opts.get_creation_mode()?,
                opts.get_flags_and_attributes(),
                ptr::null_mut(),
            )
        };
        if handle == c::INVALID_HANDLE_VALUE {
            Err(Error::last_os_error())
        } else {
            unsafe { Ok(File { handle: Handle::from_raw_handle(handle) }) }
        }
    }

    pub fn fsync(&self) -> io::Result<()> {
        cvt(unsafe { c::FlushFileBuffers(self.handle.as_raw_handle()) })?;
        Ok(())
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fsync()
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        let mut info = c::FILE_END_OF_FILE_INFO { EndOfFile: size as c::LARGE_INTEGER };
        let size = mem::size_of_val(&info);
        cvt(unsafe {
            c::SetFileInformationByHandle(
                self.handle.as_raw_handle(),
                c::FileEndOfFileInfo,
                &mut info as *mut _ as *mut _,
                size as c::DWORD,
            )
        })?;
        Ok(())
    }

    #[cfg(not(target_vendor = "uwp"))]
    pub fn file_attr(&self) -> io::Result<FileAttr> {
        unsafe {
            let mut info: c::BY_HANDLE_FILE_INFORMATION = mem::zeroed();
            cvt(c::GetFileInformationByHandle(self.handle.as_raw_handle(), &mut info))?;
            let mut reparse_tag = 0;
            if info.dwFileAttributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                let mut b = [0; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
                if let Ok((_, buf)) = self.reparse_point(&mut b) {
                    reparse_tag = buf.ReparseTag;
                }
            }
            Ok(FileAttr {
                attributes: info.dwFileAttributes,
                creation_time: info.ftCreationTime,
                last_access_time: info.ftLastAccessTime,
                last_write_time: info.ftLastWriteTime,
                file_size: (info.nFileSizeLow as u64) | ((info.nFileSizeHigh as u64) << 32),
                reparse_tag,
                volume_serial_number: Some(info.dwVolumeSerialNumber),
                number_of_links: Some(info.nNumberOfLinks),
                file_index: Some(
                    (info.nFileIndexLow as u64) | ((info.nFileIndexHigh as u64) << 32),
                ),
            })
        }
    }

    #[cfg(target_vendor = "uwp")]
    pub fn file_attr(&self) -> io::Result<FileAttr> {
        unsafe {
            let mut info: c::FILE_BASIC_INFO = mem::zeroed();
            let size = mem::size_of_val(&info);
            cvt(c::GetFileInformationByHandleEx(
                self.handle.as_raw_handle(),
                c::FileBasicInfo,
                &mut info as *mut _ as *mut libc::c_void,
                size as c::DWORD,
            ))?;
            let mut attr = FileAttr {
                attributes: info.FileAttributes,
                creation_time: c::FILETIME {
                    dwLowDateTime: info.CreationTime as c::DWORD,
                    dwHighDateTime: (info.CreationTime >> 32) as c::DWORD,
                },
                last_access_time: c::FILETIME {
                    dwLowDateTime: info.LastAccessTime as c::DWORD,
                    dwHighDateTime: (info.LastAccessTime >> 32) as c::DWORD,
                },
                last_write_time: c::FILETIME {
                    dwLowDateTime: info.LastWriteTime as c::DWORD,
                    dwHighDateTime: (info.LastWriteTime >> 32) as c::DWORD,
                },
                file_size: 0,
                reparse_tag: 0,
                volume_serial_number: None,
                number_of_links: None,
                file_index: None,
            };
            let mut info: c::FILE_STANDARD_INFO = mem::zeroed();
            let size = mem::size_of_val(&info);
            cvt(c::GetFileInformationByHandleEx(
                self.handle.as_raw_handle(),
                c::FileStandardInfo,
                &mut info as *mut _ as *mut libc::c_void,
                size as c::DWORD,
            ))?;
            attr.file_size = info.AllocationSize as u64;
            attr.number_of_links = Some(info.NumberOfLinks);
            if attr.file_type().is_reparse_point() {
                let mut b = [0; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
                if let Ok((_, buf)) = self.reparse_point(&mut b) {
                    attr.reparse_tag = buf.ReparseTag;
                }
            }
            Ok(attr)
        }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.handle.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.handle.read_vectored(bufs)
    }

    #[inline]
    pub fn is_read_vectored(&self) -> bool {
        self.handle.is_read_vectored()
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        self.handle.read_at(buf, offset)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.handle.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.handle.write_vectored(bufs)
    }

    #[inline]
    pub fn is_write_vectored(&self) -> bool {
        self.handle.is_write_vectored()
    }

    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        self.handle.write_at(buf, offset)
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, pos) = match pos {
            // Casting to `i64` is fine, `SetFilePointerEx` reinterprets this
            // integer as `u64`.
            SeekFrom::Start(n) => (c::FILE_BEGIN, n as i64),
            SeekFrom::End(n) => (c::FILE_END, n),
            SeekFrom::Current(n) => (c::FILE_CURRENT, n),
        };
        let pos = pos as c::LARGE_INTEGER;
        let mut newpos = 0;
        cvt(unsafe { c::SetFilePointerEx(self.handle.as_raw_handle(), pos, &mut newpos, whence) })?;
        Ok(newpos as u64)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        Ok(File { handle: self.handle.duplicate(0, false, c::DUPLICATE_SAME_ACCESS)? })
    }

    fn reparse_point<'a>(
        &self,
        space: &'a mut [u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE],
    ) -> io::Result<(c::DWORD, &'a c::REPARSE_DATA_BUFFER)> {
        unsafe {
            let mut bytes = 0;
            cvt({
                c::DeviceIoControl(
                    self.handle.as_raw_handle(),
                    c::FSCTL_GET_REPARSE_POINT,
                    ptr::null_mut(),
                    0,
                    space.as_mut_ptr() as *mut _,
                    space.len() as c::DWORD,
                    &mut bytes,
                    ptr::null_mut(),
                )
            })?;
            Ok((bytes, &*(space.as_ptr() as *const c::REPARSE_DATA_BUFFER)))
        }
    }

    fn readlink(&self) -> io::Result<PathBuf> {
        let mut space = [0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
        let (_bytes, buf) = self.reparse_point(&mut space)?;
        unsafe {
            let (path_buffer, subst_off, subst_len, relative) = match buf.ReparseTag {
                c::IO_REPARSE_TAG_SYMLINK => {
                    let info: *const c::SYMBOLIC_LINK_REPARSE_BUFFER =
                        &buf.rest as *const _ as *const _;
                    (
                        &(*info).PathBuffer as *const _ as *const u16,
                        (*info).SubstituteNameOffset / 2,
                        (*info).SubstituteNameLength / 2,
                        (*info).Flags & c::SYMLINK_FLAG_RELATIVE != 0,
                    )
                }
                c::IO_REPARSE_TAG_MOUNT_POINT => {
                    let info: *const c::MOUNT_POINT_REPARSE_BUFFER =
                        &buf.rest as *const _ as *const _;
                    (
                        &(*info).PathBuffer as *const _ as *const u16,
                        (*info).SubstituteNameOffset / 2,
                        (*info).SubstituteNameLength / 2,
                        false,
                    )
                }
                _ => {
                    return Err(io::Error::new_const(
                        io::ErrorKind::Uncategorized,
                        &"Unsupported reparse point type",
                    ));
                }
            };
            let subst_ptr = path_buffer.offset(subst_off as isize);
            let mut subst = slice::from_raw_parts(subst_ptr, subst_len as usize);
            // Absolute paths start with an NT internal namespace prefix `\??\`
            // We should not let it leak through.
            if !relative && subst.starts_with(&[92u16, 63u16, 63u16, 92u16]) {
                subst = &subst[4..];
            }
            Ok(PathBuf::from(OsString::from_wide(subst)))
        }
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        let mut info = c::FILE_BASIC_INFO {
            CreationTime: 0,
            LastAccessTime: 0,
            LastWriteTime: 0,
            ChangeTime: 0,
            FileAttributes: perm.attrs,
        };
        let size = mem::size_of_val(&info);
        cvt(unsafe {
            c::SetFileInformationByHandle(
                self.handle.as_raw_handle(),
                c::FileBasicInfo,
                &mut info as *mut _ as *mut _,
                size as c::DWORD,
            )
        })?;
        Ok(())
    }
    /// Get only basic file information such as attributes and file times.
    fn basic_info(&self) -> io::Result<c::FILE_BASIC_INFO> {
        unsafe {
            let mut info: c::FILE_BASIC_INFO = mem::zeroed();
            let size = mem::size_of_val(&info);
            cvt(c::GetFileInformationByHandleEx(
                self.handle.as_raw_handle(),
                c::FileBasicInfo,
                &mut info as *mut _ as *mut libc::c_void,
                size as c::DWORD,
            ))?;
            Ok(info)
        }
    }
    /// Delete using POSIX semantics.
    ///
    /// Files will be deleted as soon as the handle is closed. This is supported
    /// for Windows 10 1607 (aka RS1) and later. However some filesystem
    /// drivers will not support it even then, e.g. FAT32.
    ///
    /// If the operation is not supported for this filesystem or OS version
    /// then errors will be `ERROR_NOT_SUPPORTED` or `ERROR_INVALID_PARAMETER`.
    fn posix_delete(&self) -> io::Result<()> {
        let mut info = c::FILE_DISPOSITION_INFO_EX {
            Flags: c::FILE_DISPOSITION_DELETE
                | c::FILE_DISPOSITION_POSIX_SEMANTICS
                | c::FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE,
        };
        let size = mem::size_of_val(&info);
        cvt(unsafe {
            c::SetFileInformationByHandle(
                self.handle.as_raw_handle(),
                c::FileDispositionInfoEx,
                &mut info as *mut _ as *mut _,
                size as c::DWORD,
            )
        })?;
        Ok(())
    }

    /// Delete a file using win32 semantics. The file won't actually be deleted
    /// until all file handles are closed. However, marking a file for deletion
    /// will prevent anyone from opening a new handle to the file.
    fn win32_delete(&self) -> io::Result<()> {
        let mut info = c::FILE_DISPOSITION_INFO { DeleteFile: c::TRUE as _ };
        let size = mem::size_of_val(&info);
        cvt(unsafe {
            c::SetFileInformationByHandle(
                self.handle.as_raw_handle(),
                c::FileDispositionInfo,
                &mut info as *mut _ as *mut _,
                size as c::DWORD,
            )
        })?;
        Ok(())
    }

    /// Fill the given buffer with as many directory entries as will fit.
    /// This will remember its position and continue from the last call unless
    /// `restart` is set to `true`.
    ///
    /// The returned bool indicates if there are more entries or not.
    /// It is an error if `self` is not a directory.
    ///
    /// # Symlinks and other reparse points
    ///
    /// On Windows a file is either a directory or a non-directory.
    /// A symlink directory is simply an empty directory with some "reparse" metadata attached.
    /// So if you open a link (not its target) and iterate the directory,
    /// you will always iterate an empty directory regardless of the target.
    fn fill_dir_buff(&self, buffer: &mut DirBuff, restart: bool) -> io::Result<bool> {
        let class =
            if restart { c::FileIdBothDirectoryRestartInfo } else { c::FileIdBothDirectoryInfo };

        unsafe {
            let result = cvt(c::GetFileInformationByHandleEx(
                self.handle.as_raw_handle(),
                class,
                buffer.as_mut_ptr().cast(),
                buffer.capacity() as _,
            ));
            match result {
                Ok(_) => Ok(true),
                Err(e) if e.raw_os_error() == Some(c::ERROR_NO_MORE_FILES as _) => Ok(false),
                Err(e) => Err(e),
            }
        }
    }
}

/// A buffer for holding directory entries.
struct DirBuff {
    buffer: Vec<u8>,
}
impl DirBuff {
    fn new() -> Self {
        const BUFFER_SIZE: usize = 1024;
        Self { buffer: vec![0_u8; BUFFER_SIZE] }
    }
    fn capacity(&self) -> usize {
        self.buffer.len()
    }
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr().cast()
    }
    /// Returns a `DirBuffIter`.
    fn iter(&self) -> DirBuffIter<'_> {
        DirBuffIter::new(self)
    }
}
impl AsRef<[u8]> for DirBuff {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

/// An iterator over entries stored in a `DirBuff`.
///
/// Currently only returns file names (UTF-16 encoded).
struct DirBuffIter<'a> {
    buffer: Option<&'a [u8]>,
    cursor: usize,
}
impl<'a> DirBuffIter<'a> {
    fn new(buffer: &'a DirBuff) -> Self {
        Self { buffer: Some(buffer.as_ref()), cursor: 0 }
    }
}
impl<'a> Iterator for DirBuffIter<'a> {
    type Item = &'a [u16];
    fn next(&mut self) -> Option<Self::Item> {
        use crate::mem::size_of;
        let buffer = &self.buffer?[self.cursor..];

        // Get the name and next entry from the buffer.
        // SAFETY: The buffer contains a `FILE_ID_BOTH_DIR_INFO` struct but the
        // last field (the file name) is unsized. So an offset has to be
        // used to get the file name slice.
        let (name, next_entry) = unsafe {
            let info = buffer.as_ptr().cast::<c::FILE_ID_BOTH_DIR_INFO>();
            let next_entry = (*info).NextEntryOffset as usize;
            let name = crate::slice::from_raw_parts(
                (*info).FileName.as_ptr().cast::<u16>(),
                (*info).FileNameLength as usize / size_of::<u16>(),
            );
            (name, next_entry)
        };

        if next_entry == 0 {
            self.buffer = None
        } else {
            self.cursor += next_entry
        }

        // Skip `.` and `..` pseudo entries.
        const DOT: u16 = b'.' as u16;
        match name {
            [DOT] | [DOT, DOT] => self.next(),
            _ => Some(name),
        }
    }
}

/// Open a link relative to the parent directory, ensure no symlinks are followed.
fn open_link_no_reparse(parent: &File, name: &[u16], access: u32) -> io::Result<File> {
    // This is implemented using the lower level `NtOpenFile` function as
    // unfortunately opening a file relative to a parent is not supported by
    // win32 functions. It is however a fundamental feature of the NT kernel.
    //
    // See https://docs.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntopenfile
    unsafe {
        let mut handle = ptr::null_mut();
        let mut io_status = c::IO_STATUS_BLOCK::default();
        let name_str = c::UNICODE_STRING::from_ref(name);
        use crate::sync::atomic::{AtomicU32, Ordering};
        // The `OBJ_DONT_REPARSE` attribute ensures that we haven't been
        // tricked into following a symlink. However, it may not be available in
        // earlier versions of Windows.
        static ATTRIBUTES: AtomicU32 = AtomicU32::new(c::OBJ_DONT_REPARSE);
        let object = c::OBJECT_ATTRIBUTES {
            ObjectName: &name_str,
            RootDirectory: parent.as_raw_handle(),
            Attributes: ATTRIBUTES.load(Ordering::Relaxed),
            ..c::OBJECT_ATTRIBUTES::default()
        };
        let status = c::NtOpenFile(
            &mut handle,
            access,
            &object,
            &mut io_status,
            c::FILE_SHARE_DELETE | c::FILE_SHARE_READ | c::FILE_SHARE_WRITE,
            // If `name` is a symlink then open the link rather than the target.
            c::FILE_OPEN_REPARSE_POINT,
        );
        // Convert an NTSTATUS to the more familiar Win32 error codes (aka "DosError")
        if c::nt_success(status) {
            Ok(File::from_raw_handle(handle))
        } else if status == c::STATUS_DELETE_PENDING {
            // We make a special exception for `STATUS_DELETE_PENDING` because
            // otherwise this will be mapped to `ERROR_ACCESS_DENIED` which is
            // very unhelpful.
            Err(io::Error::from_raw_os_error(c::ERROR_DELETE_PENDING as _))
        } else if status == c::STATUS_INVALID_PARAMETER
            && ATTRIBUTES.load(Ordering::Relaxed) == c::OBJ_DONT_REPARSE
        {
            // Try without `OBJ_DONT_REPARSE`. See above.
            ATTRIBUTES.store(0, Ordering::Relaxed);
            open_link_no_reparse(parent, name, access)
        } else {
            Err(io::Error::from_raw_os_error(c::RtlNtStatusToDosError(status) as _))
        }
    }
}

impl AsInner<Handle> for File {
    fn as_inner(&self) -> &Handle {
        &self.handle
    }
}

impl IntoInner<Handle> for File {
    fn into_inner(self) -> Handle {
        self.handle
    }
}

impl FromInner<Handle> for File {
    fn from_inner(handle: Handle) -> File {
        File { handle }
    }
}

impl AsHandle for File {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.as_inner().as_handle()
    }
}

impl AsRawHandle for File {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().as_raw_handle()
    }
}

impl IntoRawHandle for File {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_raw_handle()
    }
}

impl FromRawHandle for File {
    unsafe fn from_raw_handle(raw_handle: RawHandle) -> Self {
        Self { handle: FromInner::from_inner(FromRawHandle::from_raw_handle(raw_handle)) }
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME(#24570): add more info here (e.g., mode)
        let mut b = f.debug_struct("File");
        b.field("handle", &self.handle.as_raw_handle());
        if let Ok(path) = get_path(&self) {
            b.field("path", &path);
        }
        b.finish()
    }
}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.file_size
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions { attrs: self.attributes }
    }

    pub fn attrs(&self) -> u32 {
        self.attributes
    }

    pub fn file_type(&self) -> FileType {
        FileType::new(self.attributes, self.reparse_tag)
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(self.last_write_time))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(self.last_access_time))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(SystemTime::from(self.creation_time))
    }

    pub fn modified_u64(&self) -> u64 {
        to_u64(&self.last_write_time)
    }

    pub fn accessed_u64(&self) -> u64 {
        to_u64(&self.last_access_time)
    }

    pub fn created_u64(&self) -> u64 {
        to_u64(&self.creation_time)
    }

    pub fn volume_serial_number(&self) -> Option<u32> {
        self.volume_serial_number
    }

    pub fn number_of_links(&self) -> Option<u32> {
        self.number_of_links
    }

    pub fn file_index(&self) -> Option<u64> {
        self.file_index
    }
}

fn to_u64(ft: &c::FILETIME) -> u64 {
    (ft.dwLowDateTime as u64) | ((ft.dwHighDateTime as u64) << 32)
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.attrs & c::FILE_ATTRIBUTE_READONLY != 0
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.attrs |= c::FILE_ATTRIBUTE_READONLY;
        } else {
            self.attrs &= !c::FILE_ATTRIBUTE_READONLY;
        }
    }
}

impl FileType {
    fn new(attrs: c::DWORD, reparse_tag: c::DWORD) -> FileType {
        FileType { attributes: attrs, reparse_tag }
    }
    pub fn is_dir(&self) -> bool {
        !self.is_symlink() && self.is_directory()
    }
    pub fn is_file(&self) -> bool {
        !self.is_symlink() && !self.is_directory()
    }
    pub fn is_symlink(&self) -> bool {
        self.is_reparse_point() && self.is_reparse_tag_name_surrogate()
    }
    pub fn is_symlink_dir(&self) -> bool {
        self.is_symlink() && self.is_directory()
    }
    pub fn is_symlink_file(&self) -> bool {
        self.is_symlink() && !self.is_directory()
    }
    fn is_directory(&self) -> bool {
        self.attributes & c::FILE_ATTRIBUTE_DIRECTORY != 0
    }
    fn is_reparse_point(&self) -> bool {
        self.attributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    fn is_reparse_tag_name_surrogate(&self) -> bool {
        self.reparse_tag & 0x20000000 != 0
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        let p = maybe_verbatim(p)?;
        cvt(unsafe { c::CreateDirectoryW(p.as_ptr(), ptr::null_mut()) })?;
        Ok(())
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let root = p.to_path_buf();
    let star = p.join("*");
    let path = maybe_verbatim(&star)?;

    unsafe {
        let mut wfd = mem::zeroed();
        let find_handle = c::FindFirstFileW(path.as_ptr(), &mut wfd);
        if find_handle != c::INVALID_HANDLE_VALUE {
            Ok(ReadDir {
                handle: FindNextFileHandle(find_handle),
                root: Arc::new(root),
                first: Some(wfd),
            })
        } else {
            Err(Error::last_os_error())
        }
    }
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let p_u16s = maybe_verbatim(p)?;
    cvt(unsafe { c::DeleteFileW(p_u16s.as_ptr()) })?;
    Ok(())
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let old = maybe_verbatim(old)?;
    let new = maybe_verbatim(new)?;
    cvt(unsafe { c::MoveFileExW(old.as_ptr(), new.as_ptr(), c::MOVEFILE_REPLACE_EXISTING) })?;
    Ok(())
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let p = maybe_verbatim(p)?;
    cvt(unsafe { c::RemoveDirectoryW(p.as_ptr()) })?;
    Ok(())
}

/// Open a file or directory without following symlinks.
fn open_link(path: &Path, access_mode: u32) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.access_mode(access_mode);
    // `FILE_FLAG_BACKUP_SEMANTICS` allows opening directories.
    // `FILE_FLAG_OPEN_REPARSE_POINT` opens a link instead of its target.
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS | c::FILE_FLAG_OPEN_REPARSE_POINT);
    File::open(path, &opts)
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let file = open_link(path, c::DELETE | c::FILE_LIST_DIRECTORY)?;

    // Test if the file is not a directory or a symlink to a directory.
    if (file.basic_info()?.FileAttributes & c::FILE_ATTRIBUTE_DIRECTORY) == 0 {
        return Err(io::Error::from_raw_os_error(c::ERROR_DIRECTORY as _));
    }
    let mut delete: fn(&File) -> io::Result<()> = File::posix_delete;
    let result = match delete(&file) {
        Err(e) if e.kind() == io::ErrorKind::DirectoryNotEmpty => {
            match remove_dir_all_recursive(&file, delete) {
                // Return unexpected errors.
                Err(e) if e.kind() != io::ErrorKind::DirectoryNotEmpty => return Err(e),
                result => result,
            }
        }
        // If POSIX delete is not supported for this filesystem then fallback to win32 delete.
        Err(e)
            if e.raw_os_error() == Some(c::ERROR_NOT_SUPPORTED as i32)
                || e.raw_os_error() == Some(c::ERROR_INVALID_PARAMETER as i32) =>
        {
            delete = File::win32_delete;
            Err(e)
        }
        result => result,
    };
    if result.is_ok() {
        Ok(())
    } else {
        // This is a fallback to make sure the directory is actually deleted.
        // Otherwise this function is prone to failing with `DirectoryNotEmpty`
        // due to possible delays between marking a file for deletion and the
        // file actually being deleted from the filesystem.
        //
        // So we retry a few times before giving up.
        for _ in 0..5 {
            match remove_dir_all_recursive(&file, delete) {
                Err(e) if e.kind() == io::ErrorKind::DirectoryNotEmpty => {}
                result => return result,
            }
        }
        // Try one last time.
        delete(&file)
    }
}

fn remove_dir_all_recursive(f: &File, delete: fn(&File) -> io::Result<()>) -> io::Result<()> {
    let mut buffer = DirBuff::new();
    let mut restart = true;
    // Fill the buffer and iterate the entries.
    while f.fill_dir_buff(&mut buffer, restart)? {
        for name in buffer.iter() {
            // Open the file without following symlinks and try deleting it.
            // We try opening will all needed permissions and if that is denied
            // fallback to opening without `FILE_LIST_DIRECTORY` permission.
            // Note `SYNCHRONIZE` permission is needed for synchronous access.
            let mut result =
                open_link_no_reparse(&f, name, c::SYNCHRONIZE | c::DELETE | c::FILE_LIST_DIRECTORY);
            if matches!(&result, Err(e) if e.kind() == io::ErrorKind::PermissionDenied) {
                result = open_link_no_reparse(&f, name, c::SYNCHRONIZE | c::DELETE);
            }
            match result {
                Ok(file) => match delete(&file) {
                    Err(e) if e.kind() == io::ErrorKind::DirectoryNotEmpty => {
                        // Iterate the directory's files.
                        // Ignore `DirectoryNotEmpty` errors here. They will be
                        // caught when `remove_dir_all` tries to delete the top
                        // level directory. It can then decide if to retry or not.
                        match remove_dir_all_recursive(&file, delete) {
                            Err(e) if e.kind() == io::ErrorKind::DirectoryNotEmpty => {}
                            result => result?,
                        }
                    }
                    result => result?,
                },
                // Ignore error if a delete is already in progress or the file
                // has already been deleted. It also ignores sharing violations
                // (where a file is locked by another process) as these are
                // usually temporary.
                Err(e)
                    if e.raw_os_error() == Some(c::ERROR_DELETE_PENDING as _)
                        || e.kind() == io::ErrorKind::NotFound
                        || e.raw_os_error() == Some(c::ERROR_SHARING_VIOLATION as _) => {}
                Err(e) => return Err(e),
            }
        }
        // Continue reading directory entries without restarting from the beginning,
        restart = false;
    }
    delete(&f)
}

pub fn readlink(path: &Path) -> io::Result<PathBuf> {
    // Open the link with no access mode, instead of generic read.
    // By default FILE_LIST_DIRECTORY is denied for the junction "C:\Documents and Settings", so
    // this is needed for a common case.
    let mut opts = OpenOptions::new();
    opts.access_mode(0);
    opts.custom_flags(c::FILE_FLAG_OPEN_REPARSE_POINT | c::FILE_FLAG_BACKUP_SEMANTICS);
    let file = File::open(&path, &opts)?;
    file.readlink()
}

pub fn symlink(original: &Path, link: &Path) -> io::Result<()> {
    symlink_inner(original, link, false)
}

pub fn symlink_inner(original: &Path, link: &Path, dir: bool) -> io::Result<()> {
    let original = to_u16s(original)?;
    let link = maybe_verbatim(link)?;
    let flags = if dir { c::SYMBOLIC_LINK_FLAG_DIRECTORY } else { 0 };
    // Formerly, symlink creation required the SeCreateSymbolicLink privilege. For the Windows 10
    // Creators Update, Microsoft loosened this to allow unprivileged symlink creation if the
    // computer is in Developer Mode, but SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE must be
    // added to dwFlags to opt into this behaviour.
    let result = cvt(unsafe {
        c::CreateSymbolicLinkW(
            link.as_ptr(),
            original.as_ptr(),
            flags | c::SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE,
        ) as c::BOOL
    });
    if let Err(err) = result {
        if err.raw_os_error() == Some(c::ERROR_INVALID_PARAMETER as i32) {
            // Older Windows objects to SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE,
            // so if we encounter ERROR_INVALID_PARAMETER, retry without that flag.
            cvt(unsafe {
                c::CreateSymbolicLinkW(link.as_ptr(), original.as_ptr(), flags) as c::BOOL
            })?;
        } else {
            return Err(err);
        }
    }
    Ok(())
}

#[cfg(not(target_vendor = "uwp"))]
pub fn link(original: &Path, link: &Path) -> io::Result<()> {
    let original = maybe_verbatim(original)?;
    let link = maybe_verbatim(link)?;
    cvt(unsafe { c::CreateHardLinkW(link.as_ptr(), original.as_ptr(), ptr::null_mut()) })?;
    Ok(())
}

#[cfg(target_vendor = "uwp")]
pub fn link(_original: &Path, _link: &Path) -> io::Result<()> {
    return Err(io::Error::new_const(
        io::ErrorKind::Unsupported,
        &"hard link are not supported on UWP",
    ));
}

pub fn stat(path: &Path) -> io::Result<FileAttr> {
    let mut opts = OpenOptions::new();
    // No read or write permissions are necessary
    opts.access_mode(0);
    // This flag is so we can open directories too
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS);
    let file = File::open(path, &opts)?;
    file.file_attr()
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    let mut opts = OpenOptions::new();
    // No read or write permissions are necessary
    opts.access_mode(0);
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS | c::FILE_FLAG_OPEN_REPARSE_POINT);
    let file = File::open(path, &opts)?;
    file.file_attr()
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    let p = maybe_verbatim(p)?;
    unsafe {
        cvt(c::SetFileAttributesW(p.as_ptr(), perm.attrs))?;
        Ok(())
    }
}

fn get_path(f: &File) -> io::Result<PathBuf> {
    super::fill_utf16_buf(
        |buf, sz| unsafe {
            c::GetFinalPathNameByHandleW(f.handle.as_raw_handle(), buf, sz, c::VOLUME_NAME_DOS)
        },
        |buf| PathBuf::from(OsString::from_wide(buf)),
    )
}

pub fn canonicalize(p: &Path) -> io::Result<PathBuf> {
    let mut opts = OpenOptions::new();
    // No read or write permissions are necessary
    opts.access_mode(0);
    // This flag is so we can open directories too
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS);
    let f = File::open(p, &opts)?;
    get_path(&f)
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    unsafe extern "system" fn callback(
        _TotalFileSize: c::LARGE_INTEGER,
        _TotalBytesTransferred: c::LARGE_INTEGER,
        _StreamSize: c::LARGE_INTEGER,
        StreamBytesTransferred: c::LARGE_INTEGER,
        dwStreamNumber: c::DWORD,
        _dwCallbackReason: c::DWORD,
        _hSourceFile: c::HANDLE,
        _hDestinationFile: c::HANDLE,
        lpData: c::LPVOID,
    ) -> c::DWORD {
        if dwStreamNumber == 1 {
            *(lpData as *mut i64) = StreamBytesTransferred;
        }
        c::PROGRESS_CONTINUE
    }
    let pfrom = maybe_verbatim(from)?;
    let pto = maybe_verbatim(to)?;
    let mut size = 0i64;
    cvt(unsafe {
        c::CopyFileExW(
            pfrom.as_ptr(),
            pto.as_ptr(),
            Some(callback),
            &mut size as *mut _ as *mut _,
            ptr::null_mut(),
            0,
        )
    })?;
    Ok(size as u64)
}

#[allow(dead_code)]
pub fn symlink_junction<P: AsRef<Path>, Q: AsRef<Path>>(
    original: P,
    junction: Q,
) -> io::Result<()> {
    symlink_junction_inner(original.as_ref(), junction.as_ref())
}

// Creating a directory junction on windows involves dealing with reparse
// points and the DeviceIoControl function, and this code is a skeleton of
// what can be found here:
//
// http://www.flexhex.com/docs/articles/hard-links.phtml
#[allow(dead_code)]
fn symlink_junction_inner(original: &Path, junction: &Path) -> io::Result<()> {
    let d = DirBuilder::new();
    d.mkdir(&junction)?;

    let mut opts = OpenOptions::new();
    opts.write(true);
    opts.custom_flags(c::FILE_FLAG_OPEN_REPARSE_POINT | c::FILE_FLAG_BACKUP_SEMANTICS);
    let f = File::open(junction, &opts)?;
    let h = f.as_inner().as_raw_handle();

    unsafe {
        let mut data = [0u8; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
        let db = data.as_mut_ptr() as *mut c::REPARSE_MOUNTPOINT_DATA_BUFFER;
        let buf = &mut (*db).ReparseTarget as *mut c::WCHAR;
        let mut i = 0;
        // FIXME: this conversion is very hacky
        let v = br"\??\";
        let v = v.iter().map(|x| *x as u16);
        for c in v.chain(original.as_os_str().encode_wide()) {
            *buf.offset(i) = c;
            i += 1;
        }
        *buf.offset(i) = 0;
        i += 1;
        (*db).ReparseTag = c::IO_REPARSE_TAG_MOUNT_POINT;
        (*db).ReparseTargetMaximumLength = (i * 2) as c::WORD;
        (*db).ReparseTargetLength = ((i - 1) * 2) as c::WORD;
        (*db).ReparseDataLength = (*db).ReparseTargetLength as c::DWORD + 12;

        let mut ret = 0;
        cvt(c::DeviceIoControl(
            h as *mut _,
            c::FSCTL_SET_REPARSE_POINT,
            data.as_ptr() as *mut _,
            (*db).ReparseDataLength + 8,
            ptr::null_mut(),
            0,
            &mut ret,
            ptr::null_mut(),
        ))
        .map(drop)
    }
}

// Try to see if a file exists but, unlike `exists`, report I/O errors.
pub fn try_exists(path: &Path) -> io::Result<bool> {
    // Open the file to ensure any symlinks are followed to their target.
    let mut opts = OpenOptions::new();
    // No read, write, etc access rights are needed.
    opts.access_mode(0);
    // Backup semantics enables opening directories as well as files.
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS);
    match File::open(path, &opts) {
        Err(e) => match e.kind() {
            // The file definitely does not exist
            io::ErrorKind::NotFound => Ok(false),

            // `ERROR_SHARING_VIOLATION` means that the file has been locked by
            // another process. This is often temporary so we simply report it
            // as the file existing.
            _ if e.raw_os_error() == Some(c::ERROR_SHARING_VIOLATION as i32) => Ok(true),

            // Other errors such as `ERROR_ACCESS_DENIED` may indicate that the
            // file exists. However, these types of errors are usually more
            // permanent so we report them here.
            _ => Err(e),
        },
        // The file was opened successfully therefore it must exist,
        Ok(_) => Ok(true),
    }
}
