use r_efi::protocols::file;

use super::helpers;
use crate::ffi::OsString;
use crate::fmt;
use crate::hash::Hash;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::os::uefi::ffi::OsStringExt;
use crate::path::{Path, PathBuf};
use crate::sys::time::SystemTime;
use crate::sys::unsupported;

const EOF_POS: u64 = u64::MAX;

pub struct File(uefi_fs::File);

#[derive(Clone)]
pub struct FileAttr {
    size: u64,
    attr: u64,
    times: FileTimes,
}

impl From<&uefi_fs::Info> for FileAttr {
    fn from(v: &uefi_fs::Info) -> Self {
        Self {
            size: v.file_size,
            attr: v.attribute,
            times: FileTimes {
                modified: SystemTime::new(v.modification_time),
                accessed: SystemTime::new(v.last_access_time),
                created: SystemTime::new(v.create_time),
            },
        }
    }
}

pub struct ReadDir(!);

pub struct DirEntry(!);

#[derive(Clone, Debug)]
pub struct OpenOptions {
    mode: u64,
    attr: u64,
    append: bool,
    truncate: bool,
    create_new: bool,
}

#[derive(Copy, Clone, Debug)]
pub struct FileTimes {
    modified: SystemTime,
    accessed: SystemTime,
    created: SystemTime,
}

impl Default for FileTimes {
    fn default() -> Self {
        Self { modified: SystemTime::ZERO, accessed: SystemTime::ZERO, created: SystemTime::ZERO }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FileType(u64);

#[derive(Debug)]
pub struct DirBuilder {}

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions(self.attr)
    }

    pub fn file_type(&self) -> FileType {
        FileType(self.attr)
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(self.times.modified)
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        Ok(self.times.accessed)
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        Ok(self.times.created)
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.0 & r_efi::protocols::file::READ_ONLY != 0
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        if readonly {
            self.0 |= r_efi::protocols::file::READ_ONLY;
        } else {
            self.0 &= !r_efi::protocols::file::READ_ONLY;
        }
    }
}

impl FileTimes {
    pub fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = t;
    }
    pub fn set_modified(&mut self, t: SystemTime) {
        self.modified = t;
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.0 & r_efi::protocols::file::DIRECTORY != 0
    }

    pub fn is_file(&self) -> bool {
        !self.is_dir()
    }

    // Symlinks are not supported
    pub fn is_symlink(&self) -> bool {
        false
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        self.0
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.0
    }

    pub fn file_name(&self) -> OsString {
        self.0
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        self.0
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        self.0
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            mode: file::MODE_READ,
            attr: 0,
            append: false,
            truncate: false,
            create_new: false,
        }
    }

    pub fn read(&mut self, read: bool) {
        if read {
            self.mode |= file::MODE_READ;
        } else {
            self.mode &= !file::MODE_READ;
        }
    }

    pub fn write(&mut self, write: bool) {
        if write {
            self.mode |= file::MODE_WRITE;
        } else {
            self.mode &= !file::MODE_WRITE;
        }
    }

    pub fn append(&mut self, append: bool) {
        self.append = append;
    }

    pub fn truncate(&mut self, truncate: bool) {
        self.truncate = truncate;
    }

    pub fn create(&mut self, create: bool) {
        if create {
            self.mode |= file::MODE_CREATE;
        } else {
            self.mode &= !file::MODE_CREATE;
        }
    }

    pub fn create_new(&mut self, create_new: bool) {
        self.create(true);
        self.create_new = create_new;
    }
}

impl File {
    const fn new(file: uefi_fs::File) -> Self {
        Self(file)
    }

    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        if opts.create_new {
            // Check if file already exists by trying to open in readonly
            let opts = OpenOptions::new();
            let temp = uefi_fs::File::from_path(path, opts.mode, opts.attr);

            if temp.is_ok() {
                return Err(io::const_io_error!(io::ErrorKind::AlreadyExists, "File exists"));
            }
        }

        let file = uefi_fs::File::from_path(path, opts.mode, opts.attr)?;
        let file = Self::new(file);

        if opts.truncate {
            file.truncate(0)?;
        }

        if opts.append {
            file.seek(SeekFrom::Start(EOF_POS))?;
        }

        Ok(file)
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        self.0.get_info().map(|info| FileAttr::from(info.as_ref()))
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.flush()
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.fsync()
    }

    pub fn truncate(&self, size: u64) -> io::Result<()> {
        let mut info: Box<uefi_fs::Info> = self.0.get_info()?;
        info.file_size = size;
        self.0.set_info(info.as_mut())
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        crate::io::default_read_vectored(|buf| self.read(buf), bufs)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        crate::io::default_write_vectored(|buf| self.write(buf), bufs)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        self.0.flush()
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let position: u64 = match pos {
            SeekFrom::Start(x) => x,
            SeekFrom::Current(x) => ((self.0.get_position()? as i64) + x) as u64,
            SeekFrom::End(x) => ((self.file_attr()?.size() as i64) + x) as u64,
        };

        self.0.set_position(position)?;
        Ok(position)
    }

    pub fn duplicate(&self) -> io::Result<File> {
        unsupported()
    }

    pub fn set_permissions(&self, perm: FilePermissions) -> io::Result<()> {
        let mut info = self.0.get_info()?;
        info.attribute = perm.0;
        self.0.set_info(&mut info)
    }

    pub fn set_times(&self, times: FileTimes) -> io::Result<()> {
        let mut info = self.0.get_info()?;

        if times.accessed != SystemTime::ZERO {
            info.last_access_time = helpers::uefi_time_from_duration(
                times.accessed.0,
                info.last_access_time.daylight,
                info.last_access_time.timezone,
            );
        }

        if times.created != SystemTime::ZERO {
            info.create_time = helpers::uefi_time_from_duration(
                times.created.0,
                info.create_time.daylight,
                info.create_time.timezone,
            );
        }

        if times.modified != SystemTime::ZERO {
            info.modification_time = helpers::uefi_time_from_duration(
                times.modified.0,
                info.modification_time.daylight,
                info.modification_time.timezone,
            );
        }

        self.0.set_info(&mut info)
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder {}
    }

    pub fn mkdir(&self, _p: &Path) -> io::Result<()> {
        unsupported()
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b = f.debug_struct("File");

        if let Ok(info) = self.0.get_info() {
            let flen = info.file_name.len();
            let fname = OsString::from_wide(&info.file_name[..(flen - 1)]);
            b.field("file_name", &fname);
        }

        b.finish()
    }
}

pub fn readdir(_p: &Path) -> io::Result<ReadDir> {
    unsupported()
}

pub fn unlink(_p: &Path) -> io::Result<()> {
    unsupported()
}

pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> {
    unsupported()
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    let f = uefi_fs::File::from_path(p, file::MODE_READ | file::MODE_WRITE, 0)?;

    let mut info = f.get_info()?;
    info.attribute = perm.0;
    f.set_info(&mut info)
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let f = uefi_fs::File::from_path(p, file::MODE_READ | file::MODE_WRITE, file::DIRECTORY)?;
    f.delete()
}

pub fn remove_dir_all(_path: &Path) -> io::Result<()> {
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
    let opts = OpenOptions::new();
    File::open(p, &opts)?.file_attr()
}

/// Same as stat since symlinks are not supported
pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    stat(p)
}

pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub fn copy(from: &Path, to: &Path) -> io::Result<u64> {
    let mut src = crate::fs::File::open(from)?;
    let mut dst = crate::fs::File::create(to)?;

    crate::io::copy(&mut src, &mut dst)
}

pub fn exists(path: &Path) -> io::Result<bool> {
    let f = crate::fs::File::open(path);
    match f {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

mod uefi_fs {
    #![allow(dead_code)]

    use r_efi::protocols::{device_path, file, loaded_image, simple_file_system};

    use super::super::helpers;
    use crate::boxed::Box;
    use crate::io;
    use crate::mem::MaybeUninit;
    use crate::path::Path;
    use crate::ptr::NonNull;
    use crate::sys::helpers::OwnedDevicePath;

    const BACKSLASH: u16 = 0x005c;
    const DOT: u16 = 0x002e;

    pub struct File(NonNull<file::Protocol>);

    impl File {
        pub const fn new(file: NonNull<file::Protocol>) -> Self {
            Self(file)
        }

        pub fn from_path(path: &Path, open_mode: u64, attr: u64) -> io::Result<Self> {
            let p = OwnedDevicePath::from_text(path.as_os_str())?;
            let (vol, mut path_remaining) = Self::open_volume_from_device_path(p.borrow())?;
            vol.open(&mut path_remaining, open_mode, attr)
        }

        fn open_volume_from_device_path(
            path: helpers::BorrowedDevicePath<'_>,
        ) -> io::Result<(Self, Box<[u16]>)> {
            let handles = match helpers::locate_handles(simple_file_system::PROTOCOL_GUID) {
                Ok(x) => x,
                Err(e) => return Err(e),
            };
            for handle in handles {
                let volume_device_path: NonNull<device_path::Protocol> =
                    match helpers::open_protocol(handle, device_path::PROTOCOL_GUID) {
                        Ok(x) => x,
                        Err(_) => continue,
                    };
                let volume_device_path = helpers::BorrowedDevicePath::new(volume_device_path);

                if let Some(left_path) = path_best_match(&volume_device_path, &path) {
                    return Ok((Self::open_volume(handle)?, left_path));
                }
            }

            Err(io::const_io_error!(io::ErrorKind::NotFound, "Volume Not Found"))
        }

        // Open volume on device_handle using SIMPLE_FILE_SYSTEM_PROTOCOL
        fn open_volume(device_handle: NonNull<crate::ffi::c_void>) -> io::Result<Self> {
            let simple_file_system_protocol = helpers::open_protocol::<simple_file_system::Protocol>(
                device_handle,
                simple_file_system::PROTOCOL_GUID,
            )?;

            let mut file_protocol: MaybeUninit<*mut file::Protocol> = MaybeUninit::uninit();
            let r = unsafe {
                ((*simple_file_system_protocol.as_ptr()).open_volume)(
                    simple_file_system_protocol.as_ptr(),
                    file_protocol.as_mut_ptr(),
                )
            };
            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            let p = NonNull::new(unsafe { file_protocol.assume_init() })
                .unwrap_or_else(|| unreachable!());
            Ok(Self::new(p))
        }

        fn open(&self, path: &mut [u16], open_mode: u64, attr: u64) -> io::Result<Self> {
            let file_ptr = self.0.as_ptr();
            let mut file_opened: MaybeUninit<*mut file::Protocol> = MaybeUninit::uninit();

            let r = unsafe {
                ((*file_ptr).open)(
                    file_ptr,
                    file_opened.as_mut_ptr(),
                    path.as_mut_ptr(),
                    open_mode,
                    attr,
                )
            };

            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            // SAFETY: file_opened is initialized successfully
            let p = NonNull::new(unsafe { file_opened.assume_init() })
                .unwrap_or_else(|| unreachable!());
            Ok(File::new(p))
        }

        pub fn delete(self) -> io::Result<()> {
            let file_ptr = self.0.as_ptr();
            let r = unsafe { ((*file_ptr).delete)(file_ptr) };

            // SAFETY: Spec says that this function will always return EFI_SUCCESS or
            // EFI_WARN_DELETE_FAILURE.
            //
            // Also, the file is closed even in case of warning
            if r.is_error() {
                unreachable!()
            }
            crate::mem::forget(self);

            Ok(())
        }

        pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
            let file_ptr = self.0.as_ptr();
            let mut buf_size = buf.len();
            let r =
                unsafe { ((*file_ptr).read)(file_ptr, &mut buf_size, buf.as_mut_ptr() as *mut _) };

            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            Ok(buf_size)
        }

        pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
            let file_ptr = self.0.as_ptr();
            let mut buf_size = buf.len();
            let r = unsafe { ((*file_ptr).write)(file_ptr, &mut buf_size, buf.as_ptr() as *mut _) };

            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            Ok(buf_size)
        }

        pub fn set_position(&self, position: u64) -> io::Result<()> {
            let file_ptr = self.0.as_ptr();
            let r = unsafe { ((*file_ptr).set_position)(file_ptr, position) };

            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            Ok(())
        }

        pub fn get_position(&self) -> io::Result<u64> {
            let file_ptr = self.0.as_ptr();
            let mut position = 0;
            let r = unsafe { ((*file_ptr).get_position)(file_ptr, &mut position) };

            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            Ok(position)
        }

        pub fn get_info(&self) -> io::Result<Box<Info>> {
            let mut buf_size = 0usize;
            match unsafe {
                Self::get_info_raw(
                    self.0.as_ptr(),
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

            let mut info: Box<Info> = Info::alloc(buf_size);

            unsafe {
                Self::get_info_raw(
                    self.0.as_ptr(),
                    file::INFO_ID,
                    &mut buf_size,
                    info.as_mut() as *mut Info as *mut _,
                )
            }?;

            Ok(info)
        }

        pub fn set_info(&self, info: &mut Info) -> io::Result<()> {
            let file_ptr = self.0.as_ptr();
            let mut info_id = file::INFO_ID;

            let r = unsafe {
                ((*file_ptr).set_info)(
                    self.0.as_ptr(),
                    &mut info_id,
                    info.size as usize,
                    info as *mut Info as *mut _,
                )
            };

            if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
        }

        pub fn flush(&self) -> io::Result<()> {
            let file_ptr = self.0.as_ptr();
            let r = unsafe { ((*file_ptr).flush)(file_ptr) };

            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            Ok(())
        }

        unsafe fn get_info_raw(
            protocol: *mut file::Protocol,
            mut info_guid: r_efi::efi::Guid,
            buf_size: &mut usize,
            buf: *mut crate::ffi::c_void,
        ) -> io::Result<()> {
            let r = unsafe { ((*protocol).get_info)(protocol, &mut info_guid, buf_size, buf) };
            if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
        }
    }

    impl Drop for File {
        fn drop(&mut self) {
            let file_ptr = self.0.as_ptr();
            let _ = unsafe { ((*self.0.as_ptr()).close)(file_ptr) };
        }
    }

    // Open the volume on the device_handle the image was loaded from.
    fn rootfs() -> io::Result<File> {
        let loaded_image_protocol: NonNull<loaded_image::Protocol> =
            helpers::image_handle_protocol(loaded_image::PROTOCOL_GUID)?;

        let device_handle = unsafe { (*loaded_image_protocol.as_ptr()).device_handle };
        let device_handle = NonNull::new(device_handle)
            .ok_or(io::const_io_error!(io::ErrorKind::Other, "Error getting Device Handle"))?;

        File::open_volume(device_handle)
    }

    #[repr(C)]
    #[derive(Debug)]
    pub struct Info {
        pub size: u64,
        pub file_size: u64,
        pub physical_size: u64,
        pub create_time: r_efi::system::Time,
        pub last_access_time: r_efi::system::Time,
        pub modification_time: r_efi::system::Time,
        pub attribute: u64,
        pub file_name: [r_efi::base::Char16],
    }

    impl Info {
        pub fn alloc(buf_size: usize) -> Box<Info> {
            unsafe {
                let buf_layout = crate::alloc::Layout::from_size_align(buf_size, 8).unwrap();
                let temp = crate::alloc::alloc(buf_layout);
                let name_size = (buf_size - crate::mem::size_of::<file::Info<0>>()) / 2;
                let temp_ptr: *mut Info = crate::ptr::from_raw_parts_mut(temp as *mut _, name_size);
                Box::from_raw(temp_ptr)
            }
        }
    }

    fn path_best_match<'a>(
        source: &helpers::BorrowedDevicePath<'a>,
        target: &helpers::BorrowedDevicePath<'a>,
    ) -> Option<Box<[u16]>> {
        let mut source_iter = source.iter().take_while(|x| !x.is_end_instance());
        let mut target_iter = target.iter().take_while(|x| !x.is_end_instance());

        loop {
            match (source_iter.next(), target_iter.next()) {
                (Some(x), Some(y)) if x == y => continue,
                (None, Some(y)) => return Some(y.to_path().to_text_raw().unwrap()),
                _ => return None,
            }
        }
    }
}
