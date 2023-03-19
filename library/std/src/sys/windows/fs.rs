use crate::os::windows::prelude::*;

use crate::borrow::Cow;
use crate::ffi::OsString;
use crate::fmt;
use crate::io::{self, BorrowedCursor, Error, IoSlice, IoSliceMut, SeekFrom};
use crate::mem::{self, MaybeUninit};
use crate::os::windows::io::{AsHandle, BorrowedHandle};
use crate::path::{Path, PathBuf};
use crate::ptr;
use crate::slice;
use crate::sync::Arc;
use crate::sys::handle::Handle;
use crate::sys::time::SystemTime;
use crate::sys::{c, cvt, Align8};
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::thread;

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

unsafe impl Send for OpenOptions {}
unsafe impl Sync for OpenOptions {}

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
    security_attributes: c::LPSECURITY_ATTRIBUTES,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    attrs: c::DWORD,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    accessed: Option<c::FILETIME>,
    modified: Option<c::FILETIME>,
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
        Ok(self.data.into())
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
            security_attributes: ptr::null_mut(),
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
        self.security_attributes = attrs;
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
                opts.security_attributes,
                opts.get_creation_mode()?,
                opts.get_flags_and_attributes(),
                ptr::null_mut(),
            )
        };
        if let Ok(handle) = handle.try_into() {
            Ok(File { handle: Handle::from_inner(handle) })
        } else {
            Err(Error::last_os_error())
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
                let mut attr_tag: c::FILE_ATTRIBUTE_TAG_INFO = mem::zeroed();
                cvt(c::GetFileInformationByHandleEx(
                    self.handle.as_raw_handle(),
                    c::FileAttributeTagInfo,
                    ptr::addr_of_mut!(attr_tag).cast(),
                    mem::size_of::<c::FILE_ATTRIBUTE_TAG_INFO>().try_into().unwrap(),
                ))?;
                if attr_tag.FileAttributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                    reparse_tag = attr_tag.ReparseTag;
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
                let mut attr_tag: c::FILE_ATTRIBUTE_TAG_INFO = mem::zeroed();
                cvt(c::GetFileInformationByHandleEx(
                    self.handle.as_raw_handle(),
                    c::FileAttributeTagInfo,
                    ptr::addr_of_mut!(attr_tag).cast(),
                    mem::size_of::<c::FILE_ATTRIBUTE_TAG_INFO>().try_into().unwrap(),
                ))?;
                if attr_tag.FileAttributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                    attr.reparse_tag = attr_tag.ReparseTag;
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

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.handle.read_buf(cursor)
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
        Ok(Self { handle: self.handle.try_clone()? })
    }

    // NB: returned pointer is derived from `space`, and has provenance to
    // match. A raw pointer is returned rather than a reference in order to
    // avoid narrowing provenance to the actual `REPARSE_DATA_BUFFER`.
    fn reparse_point(
        &self,
        space: &mut Align8<[MaybeUninit<u8>]>,
    ) -> io::Result<(c::DWORD, *const c::REPARSE_DATA_BUFFER)> {
        unsafe {
            let mut bytes = 0;
            cvt({
                // Grab this in advance to avoid it invalidating the pointer
                // we get from `space.0.as_mut_ptr()`.
                let len = space.0.len();
                c::DeviceIoControl(
                    self.handle.as_raw_handle(),
                    c::FSCTL_GET_REPARSE_POINT,
                    ptr::null_mut(),
                    0,
                    space.0.as_mut_ptr().cast(),
                    len as c::DWORD,
                    &mut bytes,
                    ptr::null_mut(),
                )
            })?;
            const _: () = assert!(core::mem::align_of::<c::REPARSE_DATA_BUFFER>() <= 8);
            Ok((bytes, space.0.as_ptr().cast::<c::REPARSE_DATA_BUFFER>()))
        }
    }

    fn readlink(&self) -> io::Result<PathBuf> {
        let mut space = Align8([MaybeUninit::<u8>::uninit(); c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE]);
        let (_bytes, buf) = self.reparse_point(&mut space)?;
        unsafe {
            let (path_buffer, subst_off, subst_len, relative) = match (*buf).ReparseTag {
                c::IO_REPARSE_TAG_SYMLINK => {
                    let info: *const c::SYMBOLIC_LINK_REPARSE_BUFFER =
                        ptr::addr_of!((*buf).rest).cast();
                    assert!(info.is_aligned());
                    (
                        ptr::addr_of!((*info).PathBuffer).cast::<u16>(),
                        (*info).SubstituteNameOffset / 2,
                        (*info).SubstituteNameLength / 2,
                        (*info).Flags & c::SYMLINK_FLAG_RELATIVE != 0,
                    )
                }
                c::IO_REPARSE_TAG_MOUNT_POINT => {
                    let info: *const c::MOUNT_POINT_REPARSE_BUFFER =
                        ptr::addr_of!((*buf).rest).cast();
                    assert!(info.is_aligned());
                    (
                        ptr::addr_of!((*info).PathBuffer).cast::<u16>(),
                        (*info).SubstituteNameOffset / 2,
                        (*info).SubstituteNameLength / 2,
                        false,
                    )
                }
                _ => {
                    return Err(io::const_io_error!(
                        io::ErrorKind::Uncategorized,
                        "Unsupported reparse point type",
                    ));
                }
            };
            let subst_ptr = path_buffer.add(subst_off.into());
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

    pub fn set_times(&self, times: FileTimes) -> io::Result<()> {
        let is_zero = |t: c::FILETIME| t.dwLowDateTime == 0 && t.dwHighDateTime == 0;
        if times.accessed.map_or(false, is_zero) || times.modified.map_or(false, is_zero) {
            return Err(io::const_io_error!(
                io::ErrorKind::InvalidInput,
                "Cannot set file timestamp to 0",
            ));
        }
        let is_max =
            |t: c::FILETIME| t.dwLowDateTime == c::DWORD::MAX && t.dwHighDateTime == c::DWORD::MAX;
        if times.accessed.map_or(false, is_max) || times.modified.map_or(false, is_max) {
            return Err(io::const_io_error!(
                io::ErrorKind::InvalidInput,
                "Cannot set file timestamp to 0xFFFF_FFFF_FFFF_FFFF",
            ));
        }
        cvt(unsafe {
            c::SetFileTime(self.as_handle(), None, times.accessed.as_ref(), times.modified.as_ref())
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
    buffer: Box<Align8<[MaybeUninit<u8>; Self::BUFFER_SIZE]>>,
}
impl DirBuff {
    const BUFFER_SIZE: usize = 1024;
    fn new() -> Self {
        Self {
            // Safety: `Align8<[MaybeUninit<u8>; N]>` does not need
            // initialization.
            buffer: unsafe { Box::new_uninit().assume_init() },
        }
    }
    fn capacity(&self) -> usize {
        self.buffer.0.len()
    }
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.0.as_mut_ptr().cast()
    }
    /// Returns a `DirBuffIter`.
    fn iter(&self) -> DirBuffIter<'_> {
        DirBuffIter::new(self)
    }
}
impl AsRef<[MaybeUninit<u8>]> for DirBuff {
    fn as_ref(&self) -> &[MaybeUninit<u8>] {
        &self.buffer.0
    }
}

/// An iterator over entries stored in a `DirBuff`.
///
/// Currently only returns file names (UTF-16 encoded).
struct DirBuffIter<'a> {
    buffer: Option<&'a [MaybeUninit<u8>]>,
    cursor: usize,
}
impl<'a> DirBuffIter<'a> {
    fn new(buffer: &'a DirBuff) -> Self {
        Self { buffer: Some(buffer.as_ref()), cursor: 0 }
    }
}
impl<'a> Iterator for DirBuffIter<'a> {
    type Item = (Cow<'a, [u16]>, bool);
    fn next(&mut self) -> Option<Self::Item> {
        use crate::mem::size_of;
        let buffer = &self.buffer?[self.cursor..];

        // Get the name and next entry from the buffer.
        // SAFETY:
        // - The buffer contains a `FILE_ID_BOTH_DIR_INFO` struct but the last
        //   field (the file name) is unsized. So an offset has to be used to
        //   get the file name slice.
        // - The OS has guaranteed initialization of the fields of
        //   `FILE_ID_BOTH_DIR_INFO` and the trailing filename (for at least
        //   `FileNameLength` bytes)
        let (name, is_directory, next_entry) = unsafe {
            let info = buffer.as_ptr().cast::<c::FILE_ID_BOTH_DIR_INFO>();
            // While this is guaranteed to be aligned in documentation for
            // https://docs.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_id_both_dir_info
            // it does not seem that reality is so kind, and assuming this
            // caused crashes in some cases (https://github.com/rust-lang/rust/issues/104530)
            // presumably, this can be blamed on buggy filesystem drivers, but who knows.
            let next_entry = ptr::addr_of!((*info).NextEntryOffset).read_unaligned() as usize;
            let length = ptr::addr_of!((*info).FileNameLength).read_unaligned() as usize;
            let attrs = ptr::addr_of!((*info).FileAttributes).read_unaligned();
            let name = from_maybe_unaligned(
                ptr::addr_of!((*info).FileName).cast::<u16>(),
                length / size_of::<u16>(),
            );
            let is_directory = (attrs & c::FILE_ATTRIBUTE_DIRECTORY) != 0;

            (name, is_directory, next_entry)
        };

        if next_entry == 0 {
            self.buffer = None
        } else {
            self.cursor += next_entry
        }

        // Skip `.` and `..` pseudo entries.
        const DOT: u16 = b'.' as u16;
        match &name[..] {
            [DOT] | [DOT, DOT] => self.next(),
            _ => Some((name, is_directory)),
        }
    }
}

unsafe fn from_maybe_unaligned<'a>(p: *const u16, len: usize) -> Cow<'a, [u16]> {
    if p.is_aligned() {
        Cow::Borrowed(crate::slice::from_raw_parts(p, len))
    } else {
        Cow::Owned((0..len).map(|i| p.add(i).read_unaligned()).collect())
    }
}

/// Open a link relative to the parent directory, ensure no symlinks are followed.
fn open_link_no_reparse(parent: &File, name: &[u16], access: u32) -> io::Result<File> {
    // This is implemented using the lower level `NtCreateFile` function as
    // unfortunately opening a file relative to a parent is not supported by
    // win32 functions. It is however a fundamental feature of the NT kernel.
    //
    // See https://docs.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntcreatefile
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
        let status = c::NtCreateFile(
            &mut handle,
            access,
            &object,
            &mut io_status,
            crate::ptr::null_mut(),
            0,
            c::FILE_SHARE_DELETE | c::FILE_SHARE_READ | c::FILE_SHARE_WRITE,
            c::FILE_OPEN,
            // If `name` is a symlink then open the link rather than the target.
            c::FILE_OPEN_REPARSE_POINT,
            crate::ptr::null_mut(),
            0,
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
impl From<c::WIN32_FIND_DATAW> for FileAttr {
    fn from(wfd: c::WIN32_FIND_DATAW) -> Self {
        FileAttr {
            attributes: wfd.dwFileAttributes,
            creation_time: wfd.ftCreationTime,
            last_access_time: wfd.ftLastAccessTime,
            last_write_time: wfd.ftLastWriteTime,
            file_size: ((wfd.nFileSizeHigh as u64) << 32) | (wfd.nFileSizeLow as u64),
            reparse_tag: if wfd.dwFileAttributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                // reserved unless this is a reparse point
                wfd.dwReserved0
            } else {
                0
            },
            volume_serial_number: None,
            number_of_links: None,
            file_index: None,
        }
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

impl FileTimes {
    pub fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = Some(t.into_inner());
    }

    pub fn set_modified(&mut self, t: SystemTime) {
        self.modified = Some(t.into_inner());
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

    match remove_dir_all_iterative(&file, File::posix_delete) {
        Err(e) => {
            if let Some(code) = e.raw_os_error() {
                match code as u32 {
                    // If POSIX delete is not supported for this filesystem then fallback to win32 delete.
                    c::ERROR_NOT_SUPPORTED
                    | c::ERROR_INVALID_FUNCTION
                    | c::ERROR_INVALID_PARAMETER => {
                        remove_dir_all_iterative(&file, File::win32_delete)
                    }
                    _ => Err(e),
                }
            } else {
                Err(e)
            }
        }
        ok => ok,
    }
}

fn remove_dir_all_iterative(f: &File, delete: fn(&File) -> io::Result<()>) -> io::Result<()> {
    // When deleting files we may loop this many times when certain error conditions occur.
    // This allows remove_dir_all to succeed when the error is temporary.
    const MAX_RETRIES: u32 = 10;

    let mut buffer = DirBuff::new();
    let mut dirlist = vec![f.duplicate()?];

    // FIXME: This is a hack so we can push to the dirlist vec after borrowing from it.
    fn copy_handle(f: &File) -> mem::ManuallyDrop<File> {
        unsafe { mem::ManuallyDrop::new(File::from_raw_handle(f.as_raw_handle())) }
    }

    let mut restart = true;
    while let Some(dir) = dirlist.last() {
        let dir = copy_handle(dir);

        // Fill the buffer and iterate the entries.
        let more_data = dir.fill_dir_buff(&mut buffer, restart)?;
        restart = false;
        for (name, is_directory) in buffer.iter() {
            if is_directory {
                let child_dir = open_link_no_reparse(
                    &dir,
                    &name,
                    c::SYNCHRONIZE | c::DELETE | c::FILE_LIST_DIRECTORY,
                )?;
                dirlist.push(child_dir);
            } else {
                for i in 1..=MAX_RETRIES {
                    let result = open_link_no_reparse(&dir, &name, c::SYNCHRONIZE | c::DELETE);
                    match result {
                        Ok(f) => delete(&f)?,
                        // Already deleted, so skip.
                        Err(e) if e.kind() == io::ErrorKind::NotFound => break,
                        // Retry a few times if the file is locked or a delete is already in progress.
                        Err(e)
                            if i < MAX_RETRIES
                                && (e.raw_os_error() == Some(c::ERROR_DELETE_PENDING as _)
                                    || e.raw_os_error()
                                        == Some(c::ERROR_SHARING_VIOLATION as _)) => {}
                        // Otherwise return the error.
                        Err(e) => return Err(e),
                    }
                    thread::yield_now();
                }
            }
        }
        // If there were no more files then delete the directory.
        if !more_data {
            if let Some(dir) = dirlist.pop() {
                // Retry deleting a few times in case we need to wait for a file to be deleted.
                for i in 1..=MAX_RETRIES {
                    let result = delete(&dir);
                    if let Err(e) = result {
                        if i == MAX_RETRIES || e.kind() != io::ErrorKind::DirectoryNotEmpty {
                            return Err(e);
                        }
                        thread::yield_now();
                    } else {
                        break;
                    }
                }
            }
        }
    }
    Ok(())
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
    return Err(io::const_io_error!(
        io::ErrorKind::Unsupported,
        "hard link are not supported on UWP",
    ));
}

pub fn stat(path: &Path) -> io::Result<FileAttr> {
    match metadata(path, ReparsePoint::Follow) {
        Err(err) if err.raw_os_error() == Some(c::ERROR_CANT_ACCESS_FILE as i32) => {
            if let Ok(attrs) = lstat(path) {
                if !attrs.file_type().is_symlink() {
                    return Ok(attrs);
                }
            }
            Err(err)
        }
        result => result,
    }
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    metadata(path, ReparsePoint::Open)
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReparsePoint {
    Follow = 0,
    Open = c::FILE_FLAG_OPEN_REPARSE_POINT,
}
impl ReparsePoint {
    fn as_flag(self) -> u32 {
        self as u32
    }
}

fn metadata(path: &Path, reparse: ReparsePoint) -> io::Result<FileAttr> {
    let mut opts = OpenOptions::new();
    // No read or write permissions are necessary
    opts.access_mode(0);
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS | reparse.as_flag());

    // Attempt to open the file normally.
    // If that fails with `ERROR_SHARING_VIOLATION` then retry using `FindFirstFileW`.
    // If the fallback fails for any reason we return the original error.
    match File::open(path, &opts) {
        Ok(file) => file.file_attr(),
        Err(e)
            if [Some(c::ERROR_SHARING_VIOLATION as _), Some(c::ERROR_ACCESS_DENIED as _)]
                .contains(&e.raw_os_error()) =>
        {
            // `ERROR_ACCESS_DENIED` is returned when the user doesn't have permission for the resource.
            // One such example is `System Volume Information` as default but can be created as well
            // `ERROR_SHARING_VIOLATION` will almost never be returned.
            // Usually if a file is locked you can still read some metadata.
            // However, there are special system files, such as
            // `C:\hiberfil.sys`, that are locked in a way that denies even that.
            unsafe {
                let path = maybe_verbatim(path)?;

                // `FindFirstFileW` accepts wildcard file names.
                // Fortunately wildcards are not valid file names and
                // `ERROR_SHARING_VIOLATION` means the file exists (but is locked)
                // therefore it's safe to assume the file name given does not
                // include wildcards.
                let mut wfd = mem::zeroed();
                let handle = c::FindFirstFileW(path.as_ptr(), &mut wfd);

                if handle == c::INVALID_HANDLE_VALUE {
                    // This can fail if the user does not have read access to the
                    // directory.
                    Err(e)
                } else {
                    // We no longer need the find handle.
                    c::FindClose(handle);

                    // `FindFirstFileW` reads the cached file information from the
                    // directory. The downside is that this metadata may be outdated.
                    let attrs = FileAttr::from(wfd);
                    if reparse == ReparsePoint::Follow && attrs.file_type().is_symlink() {
                        Err(e)
                    } else {
                        Ok(attrs)
                    }
                }
            }
        }
        Err(e) => Err(e),
    }
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
        let mut data = Align8([MaybeUninit::<u8>::uninit(); c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE]);
        let data_ptr = data.0.as_mut_ptr();
        let db = data_ptr.cast::<c::REPARSE_MOUNTPOINT_DATA_BUFFER>();
        // Zero the header to ensure it's fully initialized, including reserved parameters.
        *db = mem::zeroed();
        let buf = ptr::addr_of_mut!((*db).ReparseTarget).cast::<c::WCHAR>();
        let mut i = 0;
        // FIXME: this conversion is very hacky
        let v = br"\??\";
        let v = v.iter().map(|x| *x as u16);
        for c in v.chain(original.as_os_str().encode_wide()) {
            *buf.add(i) = c;
            i += 1;
        }
        *buf.add(i) = 0;
        i += 1;
        (*db).ReparseTag = c::IO_REPARSE_TAG_MOUNT_POINT;
        (*db).ReparseTargetMaximumLength = (i * 2) as c::WORD;
        (*db).ReparseTargetLength = ((i - 1) * 2) as c::WORD;
        (*db).ReparseDataLength = (*db).ReparseTargetLength as c::DWORD + 12;

        let mut ret = 0;
        cvt(c::DeviceIoControl(
            h as *mut _,
            c::FSCTL_SET_REPARSE_POINT,
            data_ptr.cast(),
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
