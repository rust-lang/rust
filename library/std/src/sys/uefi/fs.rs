//! Implemented using File Protocol

use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::hash::Hash;
use crate::io::{self, IoSlice, IoSliceMut, ReadBuf, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sys::time::SystemTime;
use crate::sys::unsupported;
use r_efi::protocols::file;

// FIXME: Do not store FileProtocol Instead store Handle
pub struct File {
    ptr: uefi_fs::FileProtocol,
    path: PathBuf,
}

#[derive(Clone, Copy, Debug)]
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
    path: PathBuf,
}

pub struct DirEntry {
    pub(crate) attr: FileAttr,
    pub(crate) name: OsString,
    path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    open_mode: u64,
    attr: u64,
    append: bool,
    truncate: bool,
    create_new: bool,
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.path, f)
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        let dir_entry = self.inner.read_dir_entry(self.path.as_path());
        if let Some(Ok(ref x)) = dir_entry {
            if x.file_name().as_os_str() == OsStr::new(".")
                || x.file_name().as_os_str() == OsStr::new("..")
            {
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
        self.path.clone()
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
        OpenOptions {
            open_mode: file::MODE_READ,
            attr: 0,
            append: false,
            truncate: false,
            create_new: false,
        }
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

    pub fn truncate(&mut self, truncate: bool) {
        self.truncate = truncate;
    }

    pub fn create(&mut self, create: bool) {
        if create {
            self.open_mode |= file::MODE_CREATE;
        } else {
            self.open_mode &= !file::MODE_CREATE;
        }
    }

    // FIXME: Should be atomic, so not sure if this is correct
    pub fn create_new(&mut self, create_new: bool) {
        self.create_new = create_new;
        self.create(true);
        self.write(true);
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        let full_path = super::path::absolute(path)?;
        let file_opened = uefi_fs::FileProtocol::from_path(path, opts.open_mode, opts.attr)?;
        let file = File { ptr: file_opened, path: full_path };
        if opts.create_new {
            if file.file_attr()?.size != 0 {
                return Err(io::Error::new(io::ErrorKind::AlreadyExists, "File already exists"));
            }
        } else if opts.truncate {
            file.truncate(0)?;
        } else if opts.append {
            // If you truncate a file, no need to seek to end
            file.seek(SeekFrom::End(0))?;
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

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        self.ptr.set_file_size(size)
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
        let position: u64 = match pos {
            SeekFrom::Start(x) => x,
            SeekFrom::Current(x) => ((self.ptr.get_position()? as i64) + x) as u64,
            SeekFrom::End(x) => ((self.file_attr()?.size as i64) + x) as u64,
        };

        self.ptr.set_position(position)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        unsupported()
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        self.ptr.set_file_attr(perm.attr)
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
        let _ = uefi_fs::FileProtocol::from_path(p, self.open_mode, self.attr)?;
        Ok(())
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b = f.debug_struct("File");
        b.field("Path", &self.path);
        b.finish()
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let abs_path = super::path::absolute(p)?;
    let open_mode = file::MODE_READ;
    let attr = file::DIRECTORY;
    let inner = uefi_fs::FileProtocol::from_path(p, open_mode, attr)?;
    Ok(ReadDir { inner, path: abs_path })
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let open_mode = file::MODE_READ | file::MODE_WRITE;
    let attr = 0;
    let file = uefi_fs::FileProtocol::from_path(p, open_mode, attr)?;
    file.delete()
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    let open_mode = file::MODE_READ | file::MODE_WRITE;
    let file = uefi_fs::FileProtocol::from_path(old, open_mode, 0)?;

    // Delete if new already exists
    if let Ok(f) = uefi_fs::FileProtocol::from_path(new, open_mode, 0) {
        f.delete()?;
    }

    file.set_file_name(new.as_os_str())
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    let open_mode = file::MODE_READ | file::MODE_WRITE;
    let file = uefi_fs::FileProtocol::from_path(p, open_mode, 0)?;
    file.set_file_attr(perm.attr)
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let open_mode = file::MODE_READ | file::MODE_WRITE;
    let attr = file::DIRECTORY;
    let file = uefi_fs::FileProtocol::from_path(p, open_mode, attr)?;
    file.delete()
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let open_mode = file::MODE_READ | file::MODE_WRITE;
    let attr = file::DIRECTORY;
    let file = uefi_fs::FileProtocol::from_path(path, open_mode, attr)?;
    cascade_delete(file, path)
}

pub fn try_exists(path: &Path) -> io::Result<bool> {
    match uefi_fs::FileProtocol::from_path(path, file::MODE_READ, 0) {
        Ok(_) => Ok(true),
        Err(e) => match e.kind() {
            io::ErrorKind::NotFound => Ok(false),
            _ => Err(e),
        },
    }
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
    let opts = OpenOptions {
        open_mode: file::MODE_READ,
        attr: 0,
        append: false,
        truncate: false,
        create_new: false,
    };
    File::open(p, &opts)?.file_attr()
}

// Shoule be same as stat since symlinks are not implemented anyway
pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    stat(p)
}

// Not sure how to implement. Tryied doing a round conversion from EFI_DEVICE_PATH protocol but
// that doesn't work either.
pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

// FIXME: Find an efficient implementation
pub fn copy(_from: &Path, _to: &Path) -> io::Result<u64> {
    unsupported()
}

// Liberal Cascade Delete
// The file should not point to root
fn cascade_delete(file: uefi_fs::FileProtocol, path: &Path) -> io::Result<()> {
    // Skip "." and ".."
    let _ = file.read_dir_entry(path);
    let _ = file.read_dir_entry(path);

    while let Some(dir_entry) = file.read_dir_entry(path) {
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
                    let _ = cascade_delete(new_file, &dir_entry.path);
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
    use crate::default::Default;
    use crate::ffi::OsStr;
    use crate::ffi::OsString;
    use crate::io;
    use crate::mem::MaybeUninit;
    use crate::os::uefi;
    use crate::os::uefi::ffi::{OsStrExt, OsStringExt};
    use crate::os::uefi::raw::VariableSizeType;
    use crate::path::{Path, PathBuf};
    use crate::ptr::NonNull;
    use r_efi::protocols::file;

    // Wrapper around File Protocol. Automatically closes file/directories on being dropped.
    #[derive(Clone)]
    pub(crate) struct FileProtocol {
        inner: NonNull<file::Protocol>,
    }

    impl FileProtocol {
        fn new(inner: NonNull<file::Protocol>) -> FileProtocol {
            FileProtocol { inner }
        }

        pub fn from_path(path: &Path, open_mode: u64, attr: u64) -> io::Result<Self> {
            match path.prefix() {
                None => {
                    let rootfs = Self::get_rootfs()?;
                    rootfs.open(path, open_mode, attr)
                }
                Some(x) => match x {
                    crate::path::Prefix::UefiDevice(prefix) => {
                        let vol = Self::get_volume_from_prefix(prefix)?;
                        let new_path: PathBuf = path
                            .components()
                            .skip_while(|p| match p {
                                crate::path::Component::Prefix(_) => true,
                                _ => false,
                            })
                            .collect();
                        vol.open(&new_path, open_mode, attr)
                    }
                    _ => Err(io::Error::new(io::ErrorKind::InvalidFilename, "Invalid File Path")),
                },
            }
        }

        fn get_rootfs() -> io::Result<FileProtocol> {
            use r_efi::protocols::loaded_image;

            let mut loaded_image_guid = loaded_image::PROTOCOL_GUID;
            let loaded_image_protocol = uefi::env::get_current_handle_protocol::<
                loaded_image::Protocol,
            >(&mut loaded_image_guid)
            .ok_or(io::Error::new(io::ErrorKind::Other, "Error getting Loaded Image Protocol"))?;

            let device_handle = unsafe { (*loaded_image_protocol.as_ptr()).device_handle };
            let device_handle = NonNull::new(device_handle)
                .ok_or(io::Error::new(io::ErrorKind::Other, "Error getting Device Handle"))?;

            Self::get_volume(device_handle)
        }

        fn get_volume_from_prefix(prefix: &OsStr) -> io::Result<Self> {
            use r_efi::protocols::{device_path, simple_file_system};

            let handles =
                match crate::os::uefi::env::locate_handles(simple_file_system::PROTOCOL_GUID) {
                    Ok(x) => x,
                    Err(e) => return Err(e),
                };
            for handle in handles {
                let volume_device_path =
                    match crate::os::uefi::env::open_protocol(handle, device_path::PROTOCOL_GUID) {
                        Ok(x) => x,
                        Err(_) => continue,
                    };

                let volume_path = match PathBuf::try_from(volume_device_path) {
                    Ok(x) => x,
                    Err(_) => continue,
                };

                if volume_path.as_os_str().bytes() == prefix.bytes().split_last().unwrap().1 {
                    return Self::get_volume(handle);
                }
            }
            Err(io::Error::new(io::ErrorKind::NotFound, "Volume Not Found"))
        }

        fn get_volume(device_handle: NonNull<crate::ffi::c_void>) -> io::Result<Self> {
            use r_efi::protocols::simple_file_system;

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
                Err(status_to_io_error(r))
            } else {
                let p = NonNull::new(unsafe { file_protocol.assume_init() })
                    .ok_or(io::Error::new(io::ErrorKind::Other, "Null Rootfs"))?;
                Ok(Self::new(p))
            }
        }

        pub fn open(&self, path: &Path, open_mode: u64, attr: u64) -> io::Result<FileProtocol> {
            let mut file_opened: MaybeUninit<*mut file::Protocol> = MaybeUninit::uninit();
            unsafe {
                Self::open_raw(
                    self.inner.as_ptr(),
                    file_opened.as_mut_ptr(),
                    path.as_os_str().to_ffi_string().as_mut_ptr(),
                    open_mode,
                    attr,
                )
            }?;
            let p = NonNull::new(unsafe { file_opened.assume_init() })
                .ok_or(io::Error::new(io::ErrorKind::Other, "File is Null"))?;
            Ok(FileProtocol::new(p))
        }

        // Only Absolute seek is supported in UEFI
        pub fn set_position(&self, pos: u64) -> io::Result<u64> {
            unsafe { Self::set_position_raw(self.inner.as_ptr(), pos) }?;
            Ok(pos)
        }

        pub fn get_position(&self) -> io::Result<u64> {
            let mut pos: u64 = 0;
            unsafe { Self::get_position_raw(self.inner.as_ptr(), &mut pos) }?;
            Ok(pos)
        }

        pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
            let mut buffer_size = buf.len();
            unsafe {
                Self::write_raw(
                    self.inner.as_ptr(),
                    &mut buffer_size,
                    buf.as_ptr() as *mut crate::ffi::c_void,
                )
            }?;
            Ok(buffer_size)
        }

        pub fn read<T>(&self, buf: &mut [T], buffer_size: &mut usize) -> io::Result<()> {
            unsafe { Self::read_raw(self.inner.as_ptr(), buffer_size, buf.as_mut_ptr().cast()) }
        }

        pub fn flush(&self) -> io::Result<()> {
            unsafe { Self::flush_raw(self.inner.as_ptr()) }
        }

        pub fn read_dir_entry(&self, base_path: &Path) -> Option<io::Result<DirEntry>> {
            let mut buf_size = 0usize;
            if let Err(e) = unsafe {
                Self::read_raw(self.inner.as_ptr(), &mut buf_size, crate::ptr::null_mut())
            } {
                match e.kind() {
                    io::ErrorKind::FileTooLarge => {}
                    _ => return Some(Err(e)),
                }
            }

            if buf_size == 0 {
                return None;
            }

            let buf: VariableSizeType<file::Info> = match VariableSizeType::from_size(buf_size) {
                Ok(x) => x,
                Err(e) => return Some(Err(e)),
            };
            if let Err(e) =
                unsafe { Self::read_raw(self.inner.as_ptr(), &mut buf_size, buf.as_ptr().cast()) }
            {
                return Some(Err(e));
            }

            let name_len: usize = (buf_size - crate::mem::size_of::<file::Info>()) >> 1;
            let name =
                unsafe { OsString::from_ffi((*buf.as_ptr()).file_name.as_mut_ptr(), name_len) };
            let attr = FileAttr::from(buf.as_ref());

            let path = base_path.join(&name);
            Some(Ok(DirEntry { attr, name, path }))
        }

        pub fn get_file_info(&self) -> io::Result<uefi::raw::VariableSizeType<file::Info>> {
            let mut buf_size = 0usize;
            match unsafe {
                Self::get_info_raw(
                    self.inner.as_ptr(),
                    file::INFO_ID,
                    &mut buf_size,
                    crate::ptr::null_mut(),
                )
            } {
                Ok(()) => unreachable!(),
                Err(e) => match e.kind() {
                    io::ErrorKind::FileTooLarge => {}
                    _ => return Err(e),
                },
            }
            let buf: uefi::raw::VariableSizeType<file::Info> =
                uefi::raw::VariableSizeType::from_size(buf_size)?;
            match unsafe {
                Self::get_info_raw(
                    self.inner.as_ptr(),
                    file::INFO_ID,
                    &mut buf_size,
                    buf.as_ptr().cast(),
                )
            } {
                Ok(()) => Ok(buf),
                Err(e) => Err(e),
            }
        }

        pub fn set_file_size(&self, file_size: u64) -> io::Result<()> {
            use r_efi::efi::Time;

            let old_info = self.get_file_info()?;
            // Update fields with new values
            unsafe {
                (*old_info.as_ptr()).file_size = file_size;
                // Pass 0 for time values. That means the time stuff will not be updated.
                (*old_info.as_ptr()).create_time = Time::default();
                (*old_info.as_ptr()).modification_time = Time::default();
                (*old_info.as_ptr()).last_access_time = Time::default()
            }
            unsafe {
                Self::set_info_raw(
                    self.inner.as_ptr(),
                    file::INFO_ID,
                    old_info.size(),
                    old_info.as_ptr().cast(),
                )
            }
        }

        pub fn set_file_attr(&self, attribute: u64) -> io::Result<()> {
            use r_efi::efi::Time;

            let old_info = self.get_file_info()?;
            // Update fields with new values
            unsafe {
                (*old_info.as_ptr()).attribute = attribute;
                // Pass 0 for time values. That means the time stuff will not be updated.
                (*old_info.as_ptr()).create_time = Time::default();
                (*old_info.as_ptr()).modification_time = Time::default();
                (*old_info.as_ptr()).last_access_time = Time::default()
            }
            unsafe {
                Self::set_info_raw(
                    self.inner.as_ptr(),
                    file::INFO_ID,
                    old_info.size(),
                    old_info.as_ptr().cast(),
                )
            }
        }

        pub fn set_file_name(&self, file_name: &OsStr) -> io::Result<()> {
            use r_efi::efi::Time;

            let file_name = file_name.to_ffi_string();
            let old_info = self.get_file_info()?;
            let new_size =
                crate::mem::size_of::<file::Info>() + crate::mem::size_of_val(&file_name);
            let new_info: VariableSizeType<file::Info> = VariableSizeType::from_size(new_size)?;
            unsafe {
                (*new_info.as_ptr()).size = new_size as u64;
                (*new_info.as_ptr()).file_size = old_info.as_ref().file_size;
                (*new_info.as_ptr()).physical_size = old_info.as_ref().physical_size;
                (*new_info.as_ptr()).create_time = Time::default();
                (*new_info.as_ptr()).modification_time = Time::default();
                (*new_info.as_ptr()).last_access_time = Time::default();
                (*new_info.as_ptr()).attribute = old_info.as_ref().attribute;
                crate::ptr::copy(
                    file_name.as_ptr(),
                    (*new_info.as_ptr()).file_name.as_mut_ptr(),
                    file_name.len(),
                );
            }
            unsafe {
                Self::set_info_raw(
                    self.inner.as_ptr(),
                    file::INFO_ID,
                    new_info.size(),
                    new_info.as_ptr().cast(),
                )
            }
        }

        pub fn delete(self) -> io::Result<()> {
            let file = crate::mem::ManuallyDrop::new(self);
            unsafe { Self::delete_raw(file.inner.as_ptr()) }
        }

        unsafe fn open_raw(
            rootfs: *mut file::Protocol,
            file_opened: *mut *mut file::Protocol,
            path: *mut u16,
            open_mode: u64,
            attr: u64,
        ) -> io::Result<()> {
            let r = unsafe { ((*rootfs).open)(rootfs, file_opened, path, open_mode, attr) };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        unsafe fn set_position_raw(protocol: *mut file::Protocol, pos: u64) -> io::Result<()> {
            let r = unsafe { ((*protocol).set_position)(protocol, pos) };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        unsafe fn get_position_raw(protocol: *mut file::Protocol, pos: *mut u64) -> io::Result<()> {
            let r = unsafe { ((*protocol).get_position)(protocol, pos) };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        unsafe fn flush_raw(protocol: *mut file::Protocol) -> io::Result<()> {
            let r = unsafe { ((*protocol).flush)(protocol) };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        unsafe fn write_raw(
            protocol: *mut file::Protocol,
            buf_size: *mut usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let r = unsafe {
                ((*protocol).write)(
                    protocol, buf_size, // FIXME: Find if write can modify the buffer
                    buf,
                )
            };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        unsafe fn read_raw(
            protocol: *mut file::Protocol,
            buf_size: *mut usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let r = unsafe { ((*protocol).read)(protocol, buf_size, buf) };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        unsafe fn get_info_raw(
            protocol: *mut file::Protocol,
            mut info_guid: r_efi::efi::Guid,
            buf_size: &mut usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let r = unsafe { ((*protocol).get_info)(protocol, &mut info_guid, buf_size, buf) };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        unsafe fn set_info_raw(
            protocol: *mut file::Protocol,
            mut info_guid: r_efi::efi::Guid,
            buf_size: usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let r = unsafe { ((*protocol).set_info)(protocol, &mut info_guid, buf_size, buf) };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
        }

        unsafe fn delete_raw(protocol: *mut file::Protocol) -> io::Result<()> {
            let r = unsafe { ((*protocol).delete)(protocol) };
            if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
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
