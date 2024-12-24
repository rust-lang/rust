use super::api::{self, WinError};
use super::{IoResult, to_u16s};
use crate::alloc::{alloc, handle_alloc_error};
use crate::borrow::Cow;
use crate::ffi::{OsStr, OsString, c_void};
use crate::io::{self, BorrowedCursor, Error, IoSlice, IoSliceMut, SeekFrom};
use crate::mem::{self, MaybeUninit};
use crate::os::windows::io::{AsHandle, BorrowedHandle};
use crate::os::windows::prelude::*;
use crate::path::{Path, PathBuf};
use crate::sync::Arc;
use crate::sys::handle::Handle;
use crate::sys::path::maybe_verbatim;
use crate::sys::time::SystemTime;
use crate::sys::{Align8, c, cvt};
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::{fmt, ptr, slice};

mod remove_dir_all;
use remove_dir_all::remove_dir_all_iterative;

pub struct File {
    handle: Handle,
}

#[derive(Clone)]
pub struct FileAttr {
    attributes: u32,
    creation_time: c::FILETIME,
    last_access_time: c::FILETIME,
    last_write_time: c::FILETIME,
    change_time: Option<c::FILETIME>,
    file_size: u64,
    reparse_tag: u32,
    volume_serial_number: Option<u32>,
    number_of_links: Option<u32>,
    file_index: Option<u64>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileType {
    attributes: u32,
    reparse_tag: u32,
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
    access_mode: Option<u32>,
    attributes: u32,
    share_mode: u32,
    security_qos_flags: u32,
    security_attributes: *mut c::SECURITY_ATTRIBUTES,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    attrs: u32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    accessed: Option<c::FILETIME>,
    modified: Option<c::FILETIME>,
    created: Option<c::FILETIME>,
}

impl fmt::Debug for c::FILETIME {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let time = ((self.dwHighDateTime as u64) << 32) | self.dwLowDateTime as u64;
        f.debug_tuple("FILETIME").field(&time).finish()
    }
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
        if self.handle.0 == c::INVALID_HANDLE_VALUE {
            // This iterator was initialized with an `INVALID_HANDLE_VALUE` as its handle.
            // Simply return `None` because this is only the case when `FindFirstFileExW` in
            // the construction of this iterator returns `ERROR_FILE_NOT_FOUND` which means
            // no matchhing files can be found.
            return None;
        }
        if let Some(first) = self.first.take() {
            if let Some(e) = DirEntry::new(&self.root, &first) {
                return Some(Ok(e));
            }
        }
        unsafe {
            let mut wfd = mem::zeroed();
            loop {
                if c::FindNextFileW(self.handle.0, &mut wfd) == 0 {
                    match api::get_last_error() {
                        WinError::NO_MORE_FILES => return None,
                        WinError { code } => {
                            return Some(Err(Error::from_raw_os_error(code as i32)));
                        }
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
        self.root.join(self.file_name())
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
    pub fn security_attributes(&mut self, attrs: *mut c::SECURITY_ATTRIBUTES) {
        self.security_attributes = attrs;
    }

    fn get_access_mode(&self) -> io::Result<u32> {
        match (self.read, self.write, self.append, self.access_mode) {
            (.., Some(mode)) => Ok(mode),
            (true, false, false, None) => Ok(c::GENERIC_READ),
            (false, true, false, None) => Ok(c::GENERIC_WRITE),
            (true, true, false, None) => Ok(c::GENERIC_READ | c::GENERIC_WRITE),
            (false, _, true, None) => Ok(c::FILE_GENERIC_WRITE & !c::FILE_WRITE_DATA),
            (true, _, true, None) => {
                Ok(c::GENERIC_READ | (c::FILE_GENERIC_WRITE & !c::FILE_WRITE_DATA))
            }
            (false, false, false, None) => {
                Err(Error::from_raw_os_error(c::ERROR_INVALID_PARAMETER as i32))
            }
        }
    }

    fn get_creation_mode(&self) -> io::Result<u32> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(Error::from_raw_os_error(c::ERROR_INVALID_PARAMETER as i32));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(Error::from_raw_os_error(c::ERROR_INVALID_PARAMETER as i32));
                }
            }
        }

        Ok(match (self.create, self.truncate, self.create_new) {
            (false, false, false) => c::OPEN_EXISTING,
            (true, false, false) => c::OPEN_ALWAYS,
            (false, true, false) => c::TRUNCATE_EXISTING,
            // `CREATE_ALWAYS` has weird semantics so we emulate it using
            // `OPEN_ALWAYS` and a manual truncation step. See #115745.
            (true, true, false) => c::OPEN_ALWAYS,
            (_, _, true) => c::CREATE_NEW,
        })
    }

    fn get_flags_and_attributes(&self) -> u32 {
        self.custom_flags
            | self.attributes
            | self.security_qos_flags
            | if self.create_new { c::FILE_FLAG_OPEN_REPARSE_POINT } else { 0 }
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let path = maybe_verbatim(path)?;
        let creation = opts.get_creation_mode()?;
        let handle = unsafe {
            c::CreateFileW(
                path.as_ptr(),
                opts.get_access_mode()?,
                opts.share_mode,
                opts.security_attributes,
                creation,
                opts.get_flags_and_attributes(),
                ptr::null_mut(),
            )
        };
        let handle = unsafe { HandleOrInvalid::from_raw_handle(handle) };
        if let Ok(handle) = OwnedHandle::try_from(handle) {
            // Manual truncation. See #115745.
            if opts.truncate
                && creation == c::OPEN_ALWAYS
                && api::get_last_error() == WinError::ALREADY_EXISTS
            {
                unsafe {
                    // This first tries `FileAllocationInfo` but falls back to
                    // `FileEndOfFileInfo` in order to support WINE.
                    // If WINE gains support for FileAllocationInfo, we should
                    // remove the fallback.
                    let alloc = c::FILE_ALLOCATION_INFO { AllocationSize: 0 };
                    let result = c::SetFileInformationByHandle(
                        handle.as_raw_handle(),
                        c::FileEndOfFileInfo,
                        (&raw const alloc).cast::<c_void>(),
                        mem::size_of::<c::FILE_ALLOCATION_INFO>() as u32,
                    );
                    if result == 0 {
                        if api::get_last_error().code != 0 {
                            panic!("FILE_ALLOCATION_INFO failed!!!");
                        }
                        let eof = c::FILE_END_OF_FILE_INFO { EndOfFile: 0 };
                        let result = c::SetFileInformationByHandle(
                            handle.as_raw_handle(),
                            c::FileEndOfFileInfo,
                            (&raw const eof).cast::<c_void>(),
                            mem::size_of::<c::FILE_END_OF_FILE_INFO>() as u32,
                        );
                        if result == 0 {
                            return Err(io::Error::last_os_error());
                        }
                    }
                }
            }
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

    fn acquire_lock(&self, flags: c::LOCK_FILE_FLAGS) -> io::Result<()> {
        unsafe {
            let mut overlapped: c::OVERLAPPED = mem::zeroed();
            let event = c::CreateEventW(ptr::null_mut(), c::FALSE, c::FALSE, ptr::null());
            if event.is_null() {
                return Err(io::Error::last_os_error());
            }
            overlapped.hEvent = event;
            let lock_result = cvt(c::LockFileEx(
                self.handle.as_raw_handle(),
                flags,
                0,
                u32::MAX,
                u32::MAX,
                &mut overlapped,
            ));

            let final_result = match lock_result {
                Ok(_) => Ok(()),
                Err(err) => {
                    if err.raw_os_error() == Some(c::ERROR_IO_PENDING as i32) {
                        // Wait for the lock to be acquired, and get the lock operation status.
                        // This can happen asynchronously, if the file handle was opened for async IO
                        let mut bytes_transferred = 0;
                        cvt(c::GetOverlappedResult(
                            self.handle.as_raw_handle(),
                            &mut overlapped,
                            &mut bytes_transferred,
                            c::TRUE,
                        ))
                        .map(|_| ())
                    } else {
                        Err(err)
                    }
                }
            };
            c::CloseHandle(overlapped.hEvent);
            final_result
        }
    }

    pub fn lock(&self) -> io::Result<()> {
        self.acquire_lock(c::LOCKFILE_EXCLUSIVE_LOCK)
    }

    pub fn lock_shared(&self) -> io::Result<()> {
        self.acquire_lock(0)
    }

    pub fn try_lock(&self) -> io::Result<bool> {
        let result = cvt(unsafe {
            let mut overlapped = mem::zeroed();
            c::LockFileEx(
                self.handle.as_raw_handle(),
                c::LOCKFILE_EXCLUSIVE_LOCK | c::LOCKFILE_FAIL_IMMEDIATELY,
                0,
                u32::MAX,
                u32::MAX,
                &mut overlapped,
            )
        });

        match result {
            Ok(_) => Ok(true),
            Err(err)
                if err.raw_os_error() == Some(c::ERROR_IO_PENDING as i32)
                    || err.raw_os_error() == Some(c::ERROR_LOCK_VIOLATION as i32) =>
            {
                Ok(false)
            }
            Err(err) => Err(err),
        }
    }

    pub fn try_lock_shared(&self) -> io::Result<bool> {
        let result = cvt(unsafe {
            let mut overlapped = mem::zeroed();
            c::LockFileEx(
                self.handle.as_raw_handle(),
                c::LOCKFILE_FAIL_IMMEDIATELY,
                0,
                u32::MAX,
                u32::MAX,
                &mut overlapped,
            )
        });

        match result {
            Ok(_) => Ok(true),
            Err(err)
                if err.raw_os_error() == Some(c::ERROR_IO_PENDING as i32)
                    || err.raw_os_error() == Some(c::ERROR_LOCK_VIOLATION as i32) =>
            {
                Ok(false)
            }
            Err(err) => Err(err),
        }
    }

    pub fn unlock(&self) -> io::Result<()> {
        // Unlock the handle twice because LockFileEx() allows a file handle to acquire
        // both an exclusive and shared lock, in which case the documentation states that:
        // "...two unlock operations are necessary to unlock the region; the first unlock operation
        // unlocks the exclusive lock, the second unlock operation unlocks the shared lock"
        cvt(unsafe { c::UnlockFile(self.handle.as_raw_handle(), 0, 0, u32::MAX, u32::MAX) })?;
        let result =
            cvt(unsafe { c::UnlockFile(self.handle.as_raw_handle(), 0, 0, u32::MAX, u32::MAX) });
        match result {
            Ok(_) => Ok(()),
            Err(err) if err.raw_os_error() == Some(c::ERROR_NOT_LOCKED as i32) => Ok(()),
            Err(err) => Err(err),
        }
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        let info = c::FILE_END_OF_FILE_INFO { EndOfFile: size as i64 };
        api::set_file_information_by_handle(self.handle.as_raw_handle(), &info).io_result()
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
                    (&raw mut attr_tag).cast(),
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
                change_time: None, // Only available in FILE_BASIC_INFO
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
                (&raw mut info) as *mut c_void,
                size as u32,
            ))?;
            let mut attr = FileAttr {
                attributes: info.FileAttributes,
                creation_time: c::FILETIME {
                    dwLowDateTime: info.CreationTime as u32,
                    dwHighDateTime: (info.CreationTime >> 32) as u32,
                },
                last_access_time: c::FILETIME {
                    dwLowDateTime: info.LastAccessTime as u32,
                    dwHighDateTime: (info.LastAccessTime >> 32) as u32,
                },
                last_write_time: c::FILETIME {
                    dwLowDateTime: info.LastWriteTime as u32,
                    dwHighDateTime: (info.LastWriteTime >> 32) as u32,
                },
                change_time: Some(c::FILETIME {
                    dwLowDateTime: info.ChangeTime as u32,
                    dwHighDateTime: (info.ChangeTime >> 32) as u32,
                }),
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
                (&raw mut info) as *mut c_void,
                size as u32,
            ))?;
            attr.file_size = info.AllocationSize as u64;
            attr.number_of_links = Some(info.NumberOfLinks);
            if attr.file_type().is_reparse_point() {
                let mut attr_tag: c::FILE_ATTRIBUTE_TAG_INFO = mem::zeroed();
                cvt(c::GetFileInformationByHandleEx(
                    self.handle.as_raw_handle(),
                    c::FileAttributeTagInfo,
                    (&raw mut attr_tag).cast(),
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
        let pos = pos as i64;
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
    ) -> io::Result<(u32, *mut c::REPARSE_DATA_BUFFER)> {
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
                    len as u32,
                    &mut bytes,
                    ptr::null_mut(),
                )
            })?;
            const _: () = assert!(core::mem::align_of::<c::REPARSE_DATA_BUFFER>() <= 8);
            Ok((bytes, space.0.as_mut_ptr().cast::<c::REPARSE_DATA_BUFFER>()))
        }
    }

    fn readlink(&self) -> io::Result<PathBuf> {
        let mut space =
            Align8([MaybeUninit::<u8>::uninit(); c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE as usize]);
        let (_bytes, buf) = self.reparse_point(&mut space)?;
        unsafe {
            let (path_buffer, subst_off, subst_len, relative) = match (*buf).ReparseTag {
                c::IO_REPARSE_TAG_SYMLINK => {
                    let info: *mut c::SYMBOLIC_LINK_REPARSE_BUFFER = (&raw mut (*buf).rest).cast();
                    assert!(info.is_aligned());
                    (
                        (&raw mut (*info).PathBuffer).cast::<u16>(),
                        (*info).SubstituteNameOffset / 2,
                        (*info).SubstituteNameLength / 2,
                        (*info).Flags & c::SYMLINK_FLAG_RELATIVE != 0,
                    )
                }
                c::IO_REPARSE_TAG_MOUNT_POINT => {
                    let info: *mut c::MOUNT_POINT_REPARSE_BUFFER = (&raw mut (*buf).rest).cast();
                    assert!(info.is_aligned());
                    (
                        (&raw mut (*info).PathBuffer).cast::<u16>(),
                        (*info).SubstituteNameOffset / 2,
                        (*info).SubstituteNameLength / 2,
                        false,
                    )
                }
                _ => {
                    return Err(io::const_error!(
                        io::ErrorKind::Uncategorized,
                        "Unsupported reparse point type",
                    ));
                }
            };
            let subst_ptr = path_buffer.add(subst_off.into());
            let subst = slice::from_raw_parts_mut(subst_ptr, subst_len as usize);
            // Absolute paths start with an NT internal namespace prefix `\??\`
            // We should not let it leak through.
            if !relative && subst.starts_with(&[92u16, 63u16, 63u16, 92u16]) {
                // Turn `\??\` into `\\?\` (a verbatim path).
                subst[1] = b'\\' as u16;
                // Attempt to convert to a more user-friendly path.
                let user = super::args::from_wide_to_user_path(
                    subst.iter().copied().chain([0]).collect(),
                )?;
                Ok(PathBuf::from(OsString::from_wide(user.strip_suffix(&[0]).unwrap_or(&user))))
            } else {
                Ok(PathBuf::from(OsString::from_wide(subst)))
            }
        }
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        let info = c::FILE_BASIC_INFO {
            CreationTime: 0,
            LastAccessTime: 0,
            LastWriteTime: 0,
            ChangeTime: 0,
            FileAttributes: perm.attrs,
        };
        api::set_file_information_by_handle(self.handle.as_raw_handle(), &info).io_result()
    }

    pub fn set_times(&self, times: FileTimes) -> io::Result<()> {
        let is_zero = |t: c::FILETIME| t.dwLowDateTime == 0 && t.dwHighDateTime == 0;
        if times.accessed.map_or(false, is_zero)
            || times.modified.map_or(false, is_zero)
            || times.created.map_or(false, is_zero)
        {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "Cannot set file timestamp to 0",
            ));
        }
        let is_max = |t: c::FILETIME| t.dwLowDateTime == u32::MAX && t.dwHighDateTime == u32::MAX;
        if times.accessed.map_or(false, is_max)
            || times.modified.map_or(false, is_max)
            || times.created.map_or(false, is_max)
        {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "Cannot set file timestamp to 0xFFFF_FFFF_FFFF_FFFF",
            ));
        }
        cvt(unsafe {
            let created =
                times.created.as_ref().map(|a| a as *const c::FILETIME).unwrap_or(ptr::null());
            let accessed =
                times.accessed.as_ref().map(|a| a as *const c::FILETIME).unwrap_or(ptr::null());
            let modified =
                times.modified.as_ref().map(|a| a as *const c::FILETIME).unwrap_or(ptr::null());
            c::SetFileTime(self.as_raw_handle(), created, accessed, modified)
        })?;
        Ok(())
    }

    /// Gets only basic file information such as attributes and file times.
    fn basic_info(&self) -> io::Result<c::FILE_BASIC_INFO> {
        unsafe {
            let mut info: c::FILE_BASIC_INFO = mem::zeroed();
            let size = mem::size_of_val(&info);
            cvt(c::GetFileInformationByHandleEx(
                self.handle.as_raw_handle(),
                c::FileBasicInfo,
                (&raw mut info) as *mut c_void,
                size as u32,
            ))?;
            Ok(info)
        }
    }

    /// Deletes the file, consuming the file handle to ensure the delete occurs
    /// as immediately as possible.
    /// This attempts to use `posix_delete` but falls back to `win32_delete`
    /// if that is not supported by the filesystem.
    #[allow(unused)]
    fn delete(self) -> Result<(), WinError> {
        // If POSIX delete is not supported for this filesystem then fallback to win32 delete.
        match self.posix_delete() {
            Err(WinError::INVALID_PARAMETER)
            | Err(WinError::NOT_SUPPORTED)
            | Err(WinError::INVALID_FUNCTION) => self.win32_delete(),
            result => result,
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
    #[allow(unused)]
    fn posix_delete(&self) -> Result<(), WinError> {
        let info = c::FILE_DISPOSITION_INFO_EX {
            Flags: c::FILE_DISPOSITION_FLAG_DELETE
                | c::FILE_DISPOSITION_FLAG_POSIX_SEMANTICS
                | c::FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE,
        };
        api::set_file_information_by_handle(self.handle.as_raw_handle(), &info)
    }

    /// Delete a file using win32 semantics. The file won't actually be deleted
    /// until all file handles are closed. However, marking a file for deletion
    /// will prevent anyone from opening a new handle to the file.
    #[allow(unused)]
    fn win32_delete(&self) -> Result<(), WinError> {
        let info = c::FILE_DISPOSITION_INFO { DeleteFile: c::TRUE as _ };
        api::set_file_information_by_handle(self.handle.as_raw_handle(), &info)
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
    #[allow(unused)]
    fn fill_dir_buff(&self, buffer: &mut DirBuff, restart: bool) -> Result<bool, WinError> {
        let class =
            if restart { c::FileIdBothDirectoryRestartInfo } else { c::FileIdBothDirectoryInfo };

        unsafe {
            let result = c::GetFileInformationByHandleEx(
                self.as_raw_handle(),
                class,
                buffer.as_mut_ptr().cast(),
                buffer.capacity() as _,
            );
            if result == 0 {
                let err = api::get_last_error();
                if err.code == c::ERROR_NO_MORE_FILES { Ok(false) } else { Err(err) }
            } else {
                Ok(true)
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
            let next_entry = (&raw const (*info).NextEntryOffset).read_unaligned() as usize;
            let length = (&raw const (*info).FileNameLength).read_unaligned() as usize;
            let attrs = (&raw const (*info).FileAttributes).read_unaligned();
            let name = from_maybe_unaligned(
                (&raw const (*info).FileName).cast::<u16>(),
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
    unsafe {
        if p.is_aligned() {
            Cow::Borrowed(crate::slice::from_raw_parts(p, len))
        } else {
            Cow::Owned((0..len).map(|i| p.add(i).read_unaligned()).collect())
        }
    }
}

impl AsInner<Handle> for File {
    #[inline]
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
        unsafe {
            Self { handle: FromInner::from_inner(FromRawHandle::from_raw_handle(raw_handle)) }
        }
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME(#24570): add more info here (e.g., mode)
        let mut b = f.debug_struct("File");
        b.field("handle", &self.handle.as_raw_handle());
        if let Ok(path) = get_path(self) {
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

    pub fn changed_u64(&self) -> Option<u64> {
        self.change_time.as_ref().map(|c| to_u64(c))
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
            change_time: None,
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

    pub fn set_created(&mut self, t: SystemTime) {
        self.created = Some(t.into_inner());
    }
}

impl FileType {
    fn new(attrs: u32, reparse_tag: u32) -> FileType {
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
    // We push a `*` to the end of the path which cause the empty path to be
    // treated as the current directory. So, for consistency with other platforms,
    // we explicitly error on the empty path.
    if p.as_os_str().is_empty() {
        // Return an error code consistent with other ways of opening files.
        // E.g. fs::metadata or File::open.
        return Err(io::Error::from_raw_os_error(c::ERROR_PATH_NOT_FOUND as i32));
    }
    let root = p.to_path_buf();
    let star = p.join("*");
    let path = maybe_verbatim(&star)?;

    unsafe {
        let mut wfd: c::WIN32_FIND_DATAW = mem::zeroed();
        // this is like FindFirstFileW (see https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-findfirstfileexw),
        // but with FindExInfoBasic it should skip filling WIN32_FIND_DATAW.cAlternateFileName
        // (see https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-win32_find_dataw)
        // (which will be always null string value and currently unused) and should be faster.
        //
        // We can pass FIND_FIRST_EX_LARGE_FETCH to dwAdditionalFlags to speed up things more,
        // but as we don't know user's use profile of this function, lets be conservative.
        let find_handle = c::FindFirstFileExW(
            path.as_ptr(),
            c::FindExInfoBasic,
            &mut wfd as *mut _ as _,
            c::FindExSearchNameMatch,
            ptr::null(),
            0,
        );

        if find_handle != c::INVALID_HANDLE_VALUE {
            Ok(ReadDir {
                handle: FindNextFileHandle(find_handle),
                root: Arc::new(root),
                first: Some(wfd),
            })
        } else {
            // The status `ERROR_FILE_NOT_FOUND` is returned by the `FindFirstFileExW` function
            // if no matching files can be found, but not necessarily that the path to find the
            // files in does not exist.
            //
            // Hence, a check for whether the path to search in exists is added when the last
            // os error returned by Windows is `ERROR_FILE_NOT_FOUND` to handle this scenario.
            // If that is the case, an empty `ReadDir` iterator is returned as it returns `None`
            // in the initial `.next()` invocation because `ERROR_NO_MORE_FILES` would have been
            // returned by the `FindNextFileW` function.
            //
            // See issue #120040: https://github.com/rust-lang/rust/issues/120040.
            let last_error = api::get_last_error();
            if last_error == WinError::FILE_NOT_FOUND {
                return Ok(ReadDir {
                    handle: FindNextFileHandle(find_handle),
                    root: Arc::new(root),
                    first: None,
                });
            }

            // Just return the error constructed from the raw OS error if the above is not the case.
            //
            // Note: `ERROR_PATH_NOT_FOUND` would have been returned by the `FindFirstFileExW` function
            // when the path to search in does not exist in the first place.
            Err(Error::from_raw_os_error(last_error.code as i32))
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

    let new_len_without_nul_in_bytes = (new.len() - 1).try_into().unwrap();

    // The last field of FILE_RENAME_INFO, the file name, is unsized,
    // and FILE_RENAME_INFO has two padding bytes.
    // Therefore we need to make sure to not allocate less than
    // size_of::<c::FILE_RENAME_INFO>() bytes, which would be the case with
    // 0 or 1 character paths + a null byte.
    let struct_size = mem::size_of::<c::FILE_RENAME_INFO>()
        .max(mem::offset_of!(c::FILE_RENAME_INFO, FileName) + new.len() * mem::size_of::<u16>());

    let struct_size: u32 = struct_size.try_into().unwrap();

    let create_file = |extra_access, extra_flags| {
        let handle = unsafe {
            HandleOrInvalid::from_raw_handle(c::CreateFileW(
                old.as_ptr(),
                c::SYNCHRONIZE | c::DELETE | extra_access,
                c::FILE_SHARE_READ | c::FILE_SHARE_WRITE | c::FILE_SHARE_DELETE,
                ptr::null(),
                c::OPEN_EXISTING,
                c::FILE_ATTRIBUTE_NORMAL | c::FILE_FLAG_BACKUP_SEMANTICS | extra_flags,
                ptr::null_mut(),
            ))
        };

        OwnedHandle::try_from(handle).map_err(|_| io::Error::last_os_error())
    };

    // The following code replicates `MoveFileEx`'s behavior as reverse-engineered from its disassembly.
    // If `old` refers to a mount point, we move it instead of the target.
    let handle = match create_file(c::FILE_READ_ATTRIBUTES, c::FILE_FLAG_OPEN_REPARSE_POINT) {
        Ok(handle) => {
            let mut file_attribute_tag_info: MaybeUninit<c::FILE_ATTRIBUTE_TAG_INFO> =
                MaybeUninit::uninit();

            let result = unsafe {
                cvt(c::GetFileInformationByHandleEx(
                    handle.as_raw_handle(),
                    c::FileAttributeTagInfo,
                    file_attribute_tag_info.as_mut_ptr().cast(),
                    mem::size_of::<c::FILE_ATTRIBUTE_TAG_INFO>().try_into().unwrap(),
                ))
            };

            if let Err(err) = result {
                if err.raw_os_error() == Some(c::ERROR_INVALID_PARAMETER as _)
                    || err.raw_os_error() == Some(c::ERROR_INVALID_FUNCTION as _)
                {
                    // `GetFileInformationByHandleEx` documents that not all underlying drivers support all file information classes.
                    // Since we know we passed the correct arguments, this means the underlying driver didn't understand our request;
                    // `MoveFileEx` proceeds by reopening the file without inhibiting reparse point behavior.
                    None
                } else {
                    Some(Err(err))
                }
            } else {
                // SAFETY: The struct has been initialized by GetFileInformationByHandleEx
                let file_attribute_tag_info = unsafe { file_attribute_tag_info.assume_init() };

                if file_attribute_tag_info.FileAttributes & c::FILE_ATTRIBUTE_REPARSE_POINT != 0
                    && file_attribute_tag_info.ReparseTag != c::IO_REPARSE_TAG_MOUNT_POINT
                {
                    // The file is not a mount point: Reopen the file without inhibiting reparse point behavior.
                    None
                } else {
                    // The file is a mount point: Don't reopen the file so that the mount point gets renamed.
                    Some(Ok(handle))
                }
            }
        }
        // The underlying driver may not support `FILE_FLAG_OPEN_REPARSE_POINT`: Retry without it.
        Err(err) if err.raw_os_error() == Some(c::ERROR_INVALID_PARAMETER as _) => None,
        Err(err) => Some(Err(err)),
    }
    .unwrap_or_else(|| create_file(0, 0))?;

    let layout = core::alloc::Layout::from_size_align(
        struct_size as _,
        mem::align_of::<c::FILE_RENAME_INFO>(),
    )
    .unwrap();

    let file_rename_info = unsafe { alloc(layout) } as *mut c::FILE_RENAME_INFO;

    if file_rename_info.is_null() {
        handle_alloc_error(layout);
    }

    // SAFETY: file_rename_info is a non-null pointer pointing to memory allocated by the global allocator.
    let mut file_rename_info = unsafe { Box::from_raw(file_rename_info) };

    // SAFETY: We have allocated enough memory for a full FILE_RENAME_INFO struct and a filename.
    unsafe {
        (&raw mut (*file_rename_info).Anonymous).write(c::FILE_RENAME_INFO_0 {
            Flags: c::FILE_RENAME_FLAG_REPLACE_IF_EXISTS | c::FILE_RENAME_FLAG_POSIX_SEMANTICS,
        });

        (&raw mut (*file_rename_info).RootDirectory).write(ptr::null_mut());
        (&raw mut (*file_rename_info).FileNameLength).write(new_len_without_nul_in_bytes);

        new.as_ptr()
            .copy_to_nonoverlapping((&raw mut (*file_rename_info).FileName) as *mut u16, new.len());
    }

    // We don't use `set_file_information_by_handle` here as `FILE_RENAME_INFO` is used for both `FileRenameInfo` and `FileRenameInfoEx`.
    let result = unsafe {
        cvt(c::SetFileInformationByHandle(
            handle.as_raw_handle(),
            c::FileRenameInfoEx,
            (&raw const *file_rename_info).cast::<c_void>(),
            struct_size,
        ))
    };

    if let Err(err) = result {
        if err.raw_os_error() == Some(c::ERROR_INVALID_PARAMETER as _) {
            // FileRenameInfoEx and FILE_RENAME_FLAG_POSIX_SEMANTICS were added in Windows 10 1607; retry with FileRenameInfo.
            file_rename_info.Anonymous.ReplaceIfExists = 1;

            cvt(unsafe {
                c::SetFileInformationByHandle(
                    handle.as_raw_handle(),
                    c::FileRenameInfo,
                    (&raw const *file_rename_info).cast::<c_void>(),
                    struct_size,
                )
            })?;
        } else {
            return Err(err);
        }
    }

    Ok(())
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let p = maybe_verbatim(p)?;
    cvt(unsafe { c::RemoveDirectoryW(p.as_ptr()) })?;
    Ok(())
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    // Open a file or directory without following symlinks.
    let mut opts = OpenOptions::new();
    opts.access_mode(c::FILE_LIST_DIRECTORY);
    // `FILE_FLAG_BACKUP_SEMANTICS` allows opening directories.
    // `FILE_FLAG_OPEN_REPARSE_POINT` opens a link instead of its target.
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS | c::FILE_FLAG_OPEN_REPARSE_POINT);
    let file = File::open(path, &opts)?;

    // Test if the file is not a directory or a symlink to a directory.
    if (file.basic_info()?.FileAttributes & c::FILE_ATTRIBUTE_DIRECTORY) == 0 {
        return Err(io::Error::from_raw_os_error(c::ERROR_DIRECTORY as _));
    }

    // Remove the directory and all its contents.
    remove_dir_all_iterative(file).io_result()
}

pub fn readlink(path: &Path) -> io::Result<PathBuf> {
    // Open the link with no access mode, instead of generic read.
    // By default FILE_LIST_DIRECTORY is denied for the junction "C:\Documents and Settings", so
    // this is needed for a common case.
    let mut opts = OpenOptions::new();
    opts.access_mode(0);
    opts.custom_flags(c::FILE_FLAG_OPEN_REPARSE_POINT | c::FILE_FLAG_BACKUP_SEMANTICS);
    let file = File::open(path, &opts)?;
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
    // added to dwFlags to opt into this behavior.
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
    return Err(
        io::const_error!(io::ErrorKind::Unsupported, "hard link are not supported on UWP",),
    );
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
    // If that fails with `ERROR_SHARING_VIOLATION` then retry using `FindFirstFileExW`.
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

                // `FindFirstFileExW` accepts wildcard file names.
                // Fortunately wildcards are not valid file names and
                // `ERROR_SHARING_VIOLATION` means the file exists (but is locked)
                // therefore it's safe to assume the file name given does not
                // include wildcards.
                let mut wfd: c::WIN32_FIND_DATAW = mem::zeroed();
                let handle = c::FindFirstFileExW(
                    path.as_ptr(),
                    c::FindExInfoBasic,
                    &mut wfd as *mut _ as _,
                    c::FindExSearchNameMatch,
                    ptr::null(),
                    0,
                );

                if handle == c::INVALID_HANDLE_VALUE {
                    // This can fail if the user does not have read access to the
                    // directory.
                    Err(e)
                } else {
                    // We no longer need the find handle.
                    c::FindClose(handle);

                    // `FindFirstFileExW` reads the cached file information from the
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
        _TotalFileSize: i64,
        _TotalBytesTransferred: i64,
        _StreamSize: i64,
        StreamBytesTransferred: i64,
        dwStreamNumber: u32,
        _dwCallbackReason: u32,
        _hSourceFile: c::HANDLE,
        _hDestinationFile: c::HANDLE,
        lpData: *const c_void,
    ) -> u32 {
        unsafe {
            if dwStreamNumber == 1 {
                *(lpData as *mut i64) = StreamBytesTransferred;
            }
            c::PROGRESS_CONTINUE
        }
    }
    let pfrom = maybe_verbatim(from)?;
    let pto = maybe_verbatim(to)?;
    let mut size = 0i64;
    cvt(unsafe {
        c::CopyFileExW(
            pfrom.as_ptr(),
            pto.as_ptr(),
            Some(callback),
            (&raw mut size) as *mut _,
            ptr::null_mut(),
            0,
        )
    })?;
    Ok(size as u64)
}

pub fn junction_point(original: &Path, link: &Path) -> io::Result<()> {
    // Create and open a new directory in one go.
    let mut opts = OpenOptions::new();
    opts.create_new(true);
    opts.write(true);
    opts.custom_flags(c::FILE_FLAG_BACKUP_SEMANTICS | c::FILE_FLAG_POSIX_SEMANTICS);
    opts.attributes(c::FILE_ATTRIBUTE_DIRECTORY);

    let d = File::open(link, &opts)?;

    // We need to get an absolute, NT-style path.
    let path_bytes = original.as_os_str().as_encoded_bytes();
    let abs_path: Vec<u16> = if path_bytes.starts_with(br"\\?\") || path_bytes.starts_with(br"\??\")
    {
        // It's already an absolute path, we just need to convert the prefix to `\??\`
        let bytes = unsafe { OsStr::from_encoded_bytes_unchecked(&path_bytes[4..]) };
        r"\??\".encode_utf16().chain(bytes.encode_wide()).collect()
    } else {
        // Get an absolute path and then convert the prefix to `\??\`
        let abs_path = crate::path::absolute(original)?.into_os_string().into_encoded_bytes();
        if abs_path.len() > 0 && abs_path[1..].starts_with(br":\") {
            let bytes = unsafe { OsStr::from_encoded_bytes_unchecked(&abs_path) };
            r"\??\".encode_utf16().chain(bytes.encode_wide()).collect()
        } else if abs_path.starts_with(br"\\.\") {
            let bytes = unsafe { OsStr::from_encoded_bytes_unchecked(&abs_path[4..]) };
            r"\??\".encode_utf16().chain(bytes.encode_wide()).collect()
        } else if abs_path.starts_with(br"\\") {
            let bytes = unsafe { OsStr::from_encoded_bytes_unchecked(&abs_path[2..]) };
            r"\??\UNC\".encode_utf16().chain(bytes.encode_wide()).collect()
        } else {
            return Err(io::const_error!(io::ErrorKind::InvalidInput, "path is not valid"));
        }
    };
    // Defined inline so we don't have to mess about with variable length buffer.
    #[repr(C)]
    pub struct MountPointBuffer {
        ReparseTag: u32,
        ReparseDataLength: u16,
        Reserved: u16,
        SubstituteNameOffset: u16,
        SubstituteNameLength: u16,
        PrintNameOffset: u16,
        PrintNameLength: u16,
        PathBuffer: [MaybeUninit<u16>; c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE as usize],
    }
    let data_len = 12 + (abs_path.len() * 2);
    if data_len > u16::MAX as usize {
        return Err(io::const_error!(io::ErrorKind::InvalidInput, "`original` path is too long"));
    }
    let data_len = data_len as u16;
    let mut header = MountPointBuffer {
        ReparseTag: c::IO_REPARSE_TAG_MOUNT_POINT,
        ReparseDataLength: data_len,
        Reserved: 0,
        SubstituteNameOffset: 0,
        SubstituteNameLength: (abs_path.len() * 2) as u16,
        PrintNameOffset: ((abs_path.len() + 1) * 2) as u16,
        PrintNameLength: 0,
        PathBuffer: [MaybeUninit::uninit(); c::MAXIMUM_REPARSE_DATA_BUFFER_SIZE as usize],
    };
    unsafe {
        let ptr = header.PathBuffer.as_mut_ptr();
        ptr.copy_from(abs_path.as_ptr().cast::<MaybeUninit<u16>>(), abs_path.len());

        let mut ret = 0;
        cvt(c::DeviceIoControl(
            d.as_raw_handle(),
            c::FSCTL_SET_REPARSE_POINT,
            (&raw const header).cast::<c_void>(),
            data_len as u32 + 8,
            ptr::null_mut(),
            0,
            &mut ret,
            ptr::null_mut(),
        ))
        .map(drop)
    }
}

// Try to see if a file exists but, unlike `exists`, report I/O errors.
pub fn exists(path: &Path) -> io::Result<bool> {
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

            // `ERROR_CANT_ACCESS_FILE` means that a file exists but that the
            // reparse point could not be handled by `CreateFile`.
            // This can happen for special files such as:
            // * Unix domain sockets which you need to `connect` to
            // * App exec links which require using `CreateProcess`
            _ if e.raw_os_error() == Some(c::ERROR_CANT_ACCESS_FILE as i32) => Ok(true),

            // Other errors such as `ERROR_ACCESS_DENIED` may indicate that the
            // file exists. However, these types of errors are usually more
            // permanent so we report them here.
            _ => Err(e),
        },
        // The file was opened successfully therefore it must exist,
        Ok(_) => Ok(true),
    }
}
