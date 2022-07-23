//! Implemented using File Protocol

use crate::ffi::OsString;
use crate::fmt;
use crate::hash::Hash;
use crate::io::{self, IoSlice, IoSliceMut, ReadBuf, SeekFrom};
// use crate::os::uefi::ffi::{OsStrExt, OsStringExt};
use crate::os::uefi;
use crate::os::uefi::raw::protocols::file;
use crate::path::{Path, PathBuf};
use crate::sys::time::SystemTime;
use crate::sys::unsupported;

// FIXME: Do not store FileProtocol Instead store Handle
pub struct File {
    ptr: uefi_fs::FileProtocol,
}

#[derive(Clone, Copy)]
pub struct FileAttr {
    size: u64,
    perm: FilePermissions,
    file_type: FileType,
    created_time: SystemTime,
    last_accessed_time: SystemTime,
    modification_time: SystemTime,
}

pub struct ReadDir {
    inner: uefi_fs::FileProtocol,
}

pub struct DirEntry {
    pub(crate) attr: FileAttr,
    pub(crate) name: OsString,
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    open_mode: u64,
    attr: u64,
    append: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FilePermissions {
    attr: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct FileType {
    attr: u64,
}

#[derive(Debug)]
pub struct DirBuilder {
    attr: u64,
    open_mode: u64,
}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn perm(&self) -> FilePermissions {
        self.perm
    }

    pub fn file_type(&self) -> FileType {
        self.file_type
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(self.modification_time)
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(self.last_accessed_time)
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(self.created_time)
    }
}

impl From<&file::Info> for FileAttr {
    fn from(info: &file::Info) -> Self {
        FileAttr {
            size: info.file_size,
            perm: FilePermissions { attr: info.attribute },
            file_type: FileType { attr: info.attribute },
            modification_time: SystemTime::from(info.modification_time),
            last_accessed_time: SystemTime::from(info.last_access_time),
            created_time: SystemTime::from(info.create_time),
        }
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.attr & file::READ_ONLY != 0
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.attr |= file::READ_ONLY;
        } else {
            self.attr &= !file::READ_ONLY;
        }
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.attr & file::DIRECTORY != 0
    }

    // Not sure if Archive is a file
    pub fn is_file(&self) -> bool {
        !self.is_dir()
    }

    // Doesn't seem like symlink can be detected/supported.
    pub fn is_symlink(&self) -> bool {
        false
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        let dir_entry = self.inner.read_dir_entry();
        if let Some(Ok(ref x)) = dir_entry {
            if x.file_name() == OsString::from(".") || x.file_name() == OsString::from("..") {
                self.next()
            } else {
                dir_entry
            }
        } else {
            dir_entry
        }
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        todo!()
    }

    pub fn file_name(&self) -> OsString {
        self.name.clone()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        Ok(self.attr)
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(self.attr.file_type())
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        // These options open file in readonly mode
        OpenOptions { open_mode: file::MODE_READ, attr: 0, append: false }
    }

    pub fn read(&mut self, read: bool) {
        if read {
            self.open_mode |= file::MODE_READ;
        } else {
            self.open_mode &= !file::MODE_READ;
        }
    }

    pub fn write(&mut self, write: bool) {
        if write {
            self.open_mode |= file::MODE_WRITE;
        } else {
            self.open_mode &= !file::MODE_WRITE;
        }
    }

    pub fn append(&mut self, append: bool) {
        self.write(true);
        self.append = append;
    }

    // FIXME: Should be possible to implement
    pub fn truncate(&mut self, _truncate: bool) {}

    pub fn create(&mut self, create: bool) {
        if create {
            self.open_mode |= file::MODE_CREATE;
        } else {
            self.open_mode &= !file::MODE_CREATE;
        }
    }

    // FIXME: Should be possible to implement
    pub fn create_new(&mut self, _create_new: bool) {}
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let rootfs = uefi_fs::FileProtocol::get_rootfs()?;
        let file_opened = rootfs.open(path, opts.open_mode, opts.attr)?;
        let file = File { ptr: file_opened };
        if opts.append {
            file.seek(SeekFrom::End(0));
        }
        Ok(file)
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        let info = self.ptr.get_file_info()?;
        Ok(FileAttr::from(info.as_ref()))
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.ptr.flush()
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fsync()
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        unimplemented!()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut buf_len = buf.len();
        self.ptr.read(buf, &mut buf_len)?;
        Ok(buf_len)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, buf: &mut ReadBuf<'_>) -> io::Result<()> {
        let mut buffer_size = buf.remaining();
        let buffer = unsafe { buf.unfilled_mut() };

        self.ptr.read(buffer, &mut buffer_size)?;

        unsafe {
            buf.assume_init(buffer_size);
        }
        buf.add_filled(buffer_size);
        Ok(())
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.ptr.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        const FILE_END: u64 = 0xFFFFFFFFFFFFFFFu64;
        let position: u64 = match pos {
            SeekFrom::Start(x) => x,
            SeekFrom::Current(x) => ((self.ptr.get_position()? as i64) + x) as u64,
            SeekFrom::End(x) => {
                // if x == 0 {
                //     FILE_END
                // } else {
                ((self.file_attr()?.size as i64) + x) as u64
                // }
            }
        };

        self.ptr.seek(position)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        unimplemented!()
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        unimplemented!()
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {
            attr: file::DIRECTORY,
            open_mode: file::MODE_READ | file::MODE_WRITE | file::MODE_CREATE,
        }
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        let rootfs = uefi_fs::FileProtocol::get_rootfs()?;
        let _ = rootfs.open(p, self.open_mode, self.attr)?;
        Ok(())
    }
}

impl fmt::Debug for File {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let open_mode = file::MODE_READ;
    let attr = file::DIRECTORY;
    let inner = {
        let rootfs = uefi_fs::FileProtocol::get_rootfs()?;
        rootfs.open(p, open_mode, attr)
    }?;
    Ok(ReadDir { inner })
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let open_mode = file::MODE_READ | file::MODE_WRITE;
    let attr = 0;
    let file = {
        let rootfs = uefi_fs::FileProtocol::get_rootfs()?;
        rootfs.open(p, open_mode, attr)
    }?;
    file.delete()
}

pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> {
    unsupported()
}

pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> {
    todo!()
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let open_mode = file::MODE_READ | file::MODE_WRITE;
    let attr = file::DIRECTORY;
    let file = {
        let rootfs = uefi_fs::FileProtocol::get_rootfs()?;
        rootfs.open(p, open_mode, attr)
    }?;
    file.delete()
}

// FIXME: Implement similar to how EFI Shell does it.
// Can be found at: ShellPkg/Library/UefiShellLevel2CommandsLib/Rm.c
// Leaving this unimplemented for now since it will need a lot of other fs stuff to be implemented
pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let open_mode = file::MODE_READ | file::MODE_WRITE;
    let attr = file::DIRECTORY;
    let file = {
        let rootfs = uefi_fs::FileProtocol::get_rootfs()?;
        rootfs.open(path, open_mode, attr)
    }?;
    cascade_delete(file)
}

pub fn try_exists(_path: &Path) -> io::Result<bool> {
    unsupported()
}

pub fn readlink(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn symlink(_original: &Path, _link: &Path) -> io::Result<()> {
    unsupported()
}

pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> {
    unsupported()
}

pub fn stat(p: &Path) -> io::Result<FileAttr> {
    let opts = OpenOptions { open_mode: file::MODE_READ, attr: 0, append: false };
    File::open(p, &opts)?.file_attr()
}

// Shoule be same as stat since symlinks are not implemented anyway
pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    stat(p)
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn copy(_from: &Path, _to: &Path) -> io::Result<u64> {
    unsupported()
}

// Liberal Cascade Delete
// The file should not point to root
fn cascade_delete(file: uefi_fs::FileProtocol) -> io::Result<()> {
    // Skip "." and ".."
    let _ = file.read_dir_entry();
    let _ = file.read_dir_entry();

    while let Some(dir_entry) = file.read_dir_entry() {
        if let Ok(dir_entry) = dir_entry {
            if let Ok(t) = dir_entry.file_type() {
                if t.is_dir() {
                    let open_mode = file::MODE_READ | file::MODE_WRITE;
                    let attr = file::DIRECTORY;
                    let new_file =
                        match file.open(&PathBuf::from(dir_entry.file_name()), open_mode, attr) {
                            Ok(x) => x,
                            Err(_) => continue,
                        };
                    let _ = cascade_delete(new_file);
                } else {
                    let open_mode = file::MODE_READ | file::MODE_WRITE;
                    let attr = 0;
                    let new_file =
                        match file.open(&PathBuf::from(dir_entry.file_name()), open_mode, attr) {
                            Ok(x) => x,
                            Err(_) => continue,
                        };
                    let _ = new_file.delete();
                }
            }
        }
    }

    file.delete()
}

mod uefi_fs {
    use super::super::common::status_to_io_error;
    use super::{DirEntry, FileAttr};
    use crate::ffi::OsString;
    use crate::io;
    use crate::mem::MaybeUninit;
    use crate::os::uefi;
    use crate::os::uefi::ffi::{OsStrExt, OsStringExt};
    use crate::os::uefi::raw::{protocols::file, Status};
    use crate::path::Path;
    use crate::ptr::NonNull;

    // Wrapper around File Protocol. Automatically closes file/directories on being dropped.
    #[derive(Clone)]
    pub(crate) struct FileProtocol {
        inner: NonNull<uefi::raw::protocols::file::Protocol>,
    }

    impl FileProtocol {
        fn new(inner: NonNull<uefi::raw::protocols::file::Protocol>) -> FileProtocol {
            FileProtocol { inner }
        }

        pub(crate) fn get_rootfs() -> io::Result<FileProtocol> {
            use uefi::raw::protocols::{loaded_image, simple_file_system};

            let mut loaded_image_guid = loaded_image::PROTOCOL_GUID;
            let loaded_image_protocol = uefi::env::get_current_handle_protocol::<
                loaded_image::Protocol,
            >(&mut loaded_image_guid)
            .ok_or(io::Error::new(io::ErrorKind::Other, "Error getting Loaded Image Protocol"))?;

            let device_handle = unsafe { (*loaded_image_protocol.as_ptr()).device_handle };
            let device_handle = NonNull::new(device_handle)
                .ok_or(io::Error::new(io::ErrorKind::Other, "Error getting Device Handle"))?;

            let mut simple_file_guid = simple_file_system::PROTOCOL_GUID;
            let simple_file_system_protocol = uefi::env::get_handle_protocol::<
                simple_file_system::Protocol,
            >(device_handle, &mut simple_file_guid)
            .ok_or(io::Error::new(io::ErrorKind::Other, "Error getting Simple File System"))?;

            let mut file_protocol: MaybeUninit<*mut file::Protocol> = MaybeUninit::uninit();
            let r = unsafe {
                ((*simple_file_system_protocol.as_ptr()).open_volume)(
                    simple_file_system_protocol.as_ptr(),
                    file_protocol.as_mut_ptr(),
                )
            };
            if r.is_error() {
                Err(io::Error::new(io::ErrorKind::Other, "Error getting rootfs"))
            } else {
                let p = NonNull::new(unsafe { file_protocol.assume_init() })
                    .ok_or(io::Error::new(io::ErrorKind::Other, "Error getting rootfs"))?;
                Ok(Self::new(p))
            }
        }

        pub(crate) fn open(
            &self,
            path: &Path,
            open_mode: u64,
            attr: u64,
        ) -> io::Result<FileProtocol> {
            let rootfs = self.inner.as_ptr();

            let mut file_opened: MaybeUninit<*mut uefi::raw::protocols::file::Protocol> =
                MaybeUninit::uninit();
            let r = unsafe {
                ((*rootfs).open)(
                    rootfs,
                    file_opened.as_mut_ptr(),
                    path.as_os_str().to_ffi_string().as_mut_ptr(),
                    open_mode,
                    attr,
                )
            };

            if r.is_error() {
                Err(status_to_io_error(&r))
            } else {
                let p = NonNull::new(unsafe { file_opened.assume_init() })
                    .ok_or(io::Error::new(io::ErrorKind::Other, "File is Null"))?;
                Ok(FileProtocol::new(p))
            }
        }

        // Only Absolute seek is supported in UEFI
        pub(crate) fn seek(&self, pos: u64) -> io::Result<u64> {
            let protocol = self.inner.as_ptr();

            let r = unsafe { ((*protocol).set_position)(protocol, pos) };

            if r.is_error() { Err(status_to_io_error(&r)) } else { Ok(pos) }
        }

        pub(crate) fn get_position(&self) -> io::Result<u64> {
            let protocol = self.inner.as_ptr();
            let mut pos: u64 = 0;

            let r = unsafe { ((*protocol).get_position)(protocol, &mut pos) };

            if r.is_error() { Err(status_to_io_error(&r)) } else { Ok(pos) }
        }

        pub(crate) fn write(&self, buf: &[u8]) -> io::Result<usize> {
            let protocol = self.inner.as_ptr();
            let mut buffer_size = buf.len();

            let r = unsafe {
                ((*protocol).write)(
                    protocol,
                    &mut buffer_size,
                    // FIXME: Find if write can modify the buffer
                    buf.as_ptr() as *mut crate::ffi::c_void,
                )
            };

            if r.is_error() { Err(status_to_io_error(&r)) } else { Ok(buffer_size) }
        }

        unsafe fn raw_read(
            protocol: *mut file::Protocol,
            buf_size: *mut usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let r = unsafe { ((*protocol).read)(protocol, buf_size, buf) };

            if r.is_error() { Err(status_to_io_error(&r)) } else { Ok(()) }
        }

        pub(crate) fn read<T>(&self, buf: &mut [T], buffer_size: &mut usize) -> io::Result<()> {
            let protocol = self.inner.as_ptr();
            unsafe { Self::raw_read(protocol, buffer_size, buf.as_mut_ptr().cast()) }
        }

        pub(crate) fn flush(&self) -> io::Result<()> {
            let protocol = self.inner.as_ptr();

            let r = unsafe { ((*protocol).flush)(protocol) };

            if r.is_error() { Err(status_to_io_error(&r)) } else { Ok(()) }
        }

        pub fn read_dir_entry(&self) -> Option<io::Result<DirEntry>> {
            use crate::alloc::{Allocator, Global, Layout};

            let protocol = self.inner.as_ptr();
            let mut buf_size = 0usize;

            match unsafe {
                Self::raw_read(self.inner.as_ptr(), &mut buf_size, crate::ptr::null_mut())
            } {
                Ok(()) => {}
                Err(e) => match e.kind() {
                    io::ErrorKind::FileTooLarge => {}
                    _ => return Some(Err(e)),
                },
            }

            if buf_size == 0 {
                return None;
            }

            let layout = match Layout::from_size_align(buf_size, 8usize) {
                Ok(x) => x,
                Err(_) => {
                    return Some(Err(io::Error::new(io::ErrorKind::Other, "Invalid buffer size")));
                }
            };

            let buf: NonNull<file::Info> = match Global.allocate(layout) {
                Ok(x) => x.cast(),
                Err(_) => {
                    return Some(Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Failed to allocate Buffer",
                    )));
                }
            };

            match unsafe { Self::raw_read(protocol, &mut buf_size, buf.as_ptr().cast()) } {
                Ok(()) => {}
                Err(e) => {
                    unsafe {
                        Global.deallocate(buf.cast(), layout);
                    }
                    return Some(Err(e));
                }
            }
            let name_len: usize = (buf_size - crate::mem::size_of::<file::Info>()) >> 1;
            let name =
                unsafe { OsString::from_ffi((*buf.as_ptr()).file_name.as_mut_ptr(), name_len) };
            let attr = FileAttr::from(unsafe { buf.as_ref() });

            unsafe {
                Global.deallocate(buf.cast(), layout);
            }

            Some(Ok(DirEntry { attr, name }))
        }

        pub fn get_file_info(&self) -> io::Result<uefi::raw::VariableSizeType<file::Info>> {
            let mut buf_size = 0usize;
            match unsafe {
                Self::get_file_info_raw(self.inner.as_ptr(), &mut buf_size, crate::ptr::null_mut())
            } {
                Ok(()) => unreachable!(),
                Err(e) => match e.kind() {
                    io::ErrorKind::FileTooLarge => {}
                    _ => return Err(e),
                },
            }
            let mut buf: uefi::raw::VariableSizeType<file::Info> =
                uefi::raw::VariableSizeType::from_size(buf_size)?;
            match unsafe {
                Self::get_file_info_raw(self.inner.as_ptr(), &mut buf_size, buf.as_ptr().cast())
            } {
                Ok(()) => Ok(buf),
                Err(e) => Err(e),
            }
        }

        unsafe fn get_file_info_raw(
            protocol: *mut file::Protocol,
            buf_size: &mut usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let mut info_guid = file::INFO_ID;
            let r = unsafe { ((*protocol).get_info)(protocol, &mut info_guid, buf_size, buf) };
            if r.is_error() { Err(status_to_io_error(&r)) } else { Ok(()) }
        }

        pub(crate) fn delete(self) -> io::Result<()> {
            let file = crate::mem::ManuallyDrop::new(self);
            let protocol = file.inner.as_ptr();
            let r = unsafe { ((*protocol).delete)(protocol) };
            if r.is_error() {
                Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Handle was closed, but the file was not deleted",
                ))
            } else {
                Ok(())
            }
        }
    }

    impl Drop for FileProtocol {
        fn drop(&mut self) {
            let protocol = self.inner.as_ptr();
            // Always returns EFI_SUCCESS
            let _ = unsafe { ((*protocol).close)(protocol) };
        }
    }
}
