use r_efi::protocols::file;

use crate::ffi::OsString;
use crate::fmt;
use crate::fs::TryLockError;
use crate::hash::Hash;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sys::pal::{helpers, unsupported};
use crate::sys::time::SystemTime;

const FILE_PERMISSIONS_MASK: u64 = r_efi::protocols::file::READ_ONLY;

pub struct File(!);

#[derive(Clone)]
pub struct FileAttr {
    attr: u64,
    size: u64,
    file_time: FileTimes,
    created: Option<SystemTime>,
}

pub struct ReadDir(uefi_fs::File);

pub struct DirEntry {
    attr: FileAttr,
    file_name: OsString,
    path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    mode: u64,
    append: bool,
    truncate: bool,
    create_new: bool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {
    accessed: Option<SystemTime>,
    modified: Option<SystemTime>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
// Bool indicates if file is readonly
pub struct FilePermissions(bool);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
// Bool indicates if directory
pub struct FileType(bool);

#[derive(Debug)]
pub struct DirBuilder;

impl FileAttr {
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions::from_attr(self.attr)
    }

    pub fn file_type(&self) -> FileType {
        FileType::from_attr(self.attr)
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        self.file_time
            .modified
            .ok_or(io::const_error!(io::ErrorKind::InvalidData, "modification time is not valid"))
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        self.file_time
            .accessed
            .ok_or(io::const_error!(io::ErrorKind::InvalidData, "last access time is not valid"))
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        self.created
            .ok_or(io::const_error!(io::ErrorKind::InvalidData, "creation time is not valid"))
    }

    fn from_uefi(info: helpers::UefiBox<file::Info>) -> Self {
        unsafe {
            Self {
                attr: (*info.as_ptr()).attribute,
                size: (*info.as_ptr()).file_size,
                file_time: FileTimes {
                    modified: uefi_fs::uefi_to_systemtime((*info.as_ptr()).modification_time),
                    accessed: uefi_fs::uefi_to_systemtime((*info.as_ptr()).last_access_time),
                },
                created: uefi_fs::uefi_to_systemtime((*info.as_ptr()).create_time),
            }
        }
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        self.0
    }

    pub fn set_readonly(&mut self, readonly: bool) {
        self.0 = readonly
    }

    const fn from_attr(attr: u64) -> Self {
        Self(attr & r_efi::protocols::file::READ_ONLY != 0)
    }

    const fn to_attr(&self) -> u64 {
        if self.0 { r_efi::protocols::file::READ_ONLY } else { 0 }
    }
}

impl FileTimes {
    pub fn set_accessed(&mut self, t: SystemTime) {
        self.accessed = Some(t);
    }

    pub fn set_modified(&mut self, t: SystemTime) {
        self.modified = Some(t);
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.0
    }

    pub fn is_file(&self) -> bool {
        !self.is_dir()
    }

    // Symlinks are not supported in UEFI
    pub fn is_symlink(&self) -> bool {
        false
    }

    const fn from_attr(attr: u64) -> Self {
        Self(attr & r_efi::protocols::file::DIRECTORY != 0)
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b = f.debug_struct("ReadDir");
        b.field("path", &self.0.path());
        b.finish()
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        match self.0.read_dir_entry() {
            Ok(None) => None,
            Ok(Some(x)) => {
                let temp = DirEntry::from_uefi(x, self.0.path());
                // Ignore "." and "..". This is how ReadDir behaves in Unix.
                if temp.file_name == "." || temp.file_name == ".." {
                    self.next()
                } else {
                    Some(Ok(temp))
                }
            }
            Err(e) => Some(Err(e)),
        }
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn file_name(&self) -> OsString {
        self.file_name.clone()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        Ok(self.attr.clone())
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(self.attr.file_type())
    }

    fn from_uefi(info: helpers::UefiBox<file::Info>, parent: &Path) -> Self {
        let file_name = uefi_fs::file_name_from_uefi(&info);
        let path = parent.join(&file_name);
        Self { file_name, path, attr: FileAttr::from_uefi(info) }
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions { mode: 0, append: false, create_new: false, truncate: false }
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
            // Valid Combinations: Read, Read/Write, Read/Write/Create
            self.read(true);
            self.mode |= file::MODE_WRITE;
        } else {
            self.mode &= !file::MODE_WRITE;
        }
    }

    pub fn append(&mut self, append: bool) {
        // Docs state that `.write(true).append(true)` has the same effect as `.append(true)`
        if append {
            self.write(true);
        }
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
        self.create_new = create_new;
    }

    #[expect(dead_code)]
    const fn is_mode_valid(&self) -> bool {
        // Valid Combinations: Read, Read/Write, Read/Write/Create
        self.mode == file::MODE_READ
            || self.mode == (file::MODE_READ | file::MODE_WRITE)
            || self.mode == (file::MODE_READ | file::MODE_WRITE | file::MODE_CREATE)
    }
}

impl File {
    pub fn open(_path: &Path, _opts: &OpenOptions) -> io::Result<File> {
        unsupported()
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        self.0
    }

    pub fn fsync(&self) -> io::Result<()> {
        self.0
    }

    pub fn datasync(&self) -> io::Result<()> {
        self.0
    }

    pub fn lock(&self) -> io::Result<()> {
        self.0
    }

    pub fn lock_shared(&self) -> io::Result<()> {
        self.0
    }

    pub fn try_lock(&self) -> Result<(), TryLockError> {
        self.0
    }

    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        self.0
    }

    pub fn unlock(&self) -> io::Result<()> {
        self.0
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        self.0
    }

    pub fn read(&self, _buf: &mut [u8]) -> io::Result<usize> {
        self.0
    }

    pub fn read_vectored(&self, _bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0
    }

    pub fn is_read_vectored(&self) -> bool {
        self.0
    }

    pub fn read_buf(&self, _cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.0
    }

    pub fn write(&self, _buf: &[u8]) -> io::Result<usize> {
        self.0
    }

    pub fn write_vectored(&self, _bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0
    }

    pub fn is_write_vectored(&self) -> bool {
        self.0
    }

    pub fn flush(&self) -> io::Result<()> {
        self.0
    }

    pub fn seek(&self, _pos: SeekFrom) -> io::Result<u64> {
        self.0
    }

    pub fn size(&self) -> Option<io::Result<u64>> {
        self.0
    }

    pub fn tell(&self) -> io::Result<u64> {
        self.0
    }

    pub fn duplicate(&self) -> io::Result<File> {
        self.0
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        self.0
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        self.0
    }
}

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        uefi_fs::mkdir(p)
    }
}

impl fmt::Debug for File {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

pub fn readdir(p: &Path) -> io::Result<ReadDir> {
    let path = crate::path::absolute(p)?;
    let f = uefi_fs::File::from_path(&path, file::MODE_READ, 0)?;
    let file_info = f.file_info()?;
    let file_attr = FileAttr::from_uefi(file_info);

    if file_attr.file_type().is_dir() {
        Ok(ReadDir(f))
    } else {
        Err(io::const_error!(io::ErrorKind::NotADirectory, "expected a directory but got a file"))
    }
}

pub fn unlink(p: &Path) -> io::Result<()> {
    let f = uefi_fs::File::from_path(p, file::MODE_READ | file::MODE_WRITE, 0)?;
    let file_info = f.file_info()?;
    let file_attr = FileAttr::from_uefi(file_info);

    if file_attr.file_type().is_file() {
        f.delete()
    } else {
        Err(io::const_error!(io::ErrorKind::IsADirectory, "expected a file but got a directory"))
    }
}

pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> {
    unsupported()
}

pub fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> {
    let f = uefi_fs::File::from_path(p, file::MODE_READ | file::MODE_WRITE, 0)?;
    let mut file_info = f.file_info()?;

    unsafe {
        (*file_info.as_mut_ptr()).attribute =
            ((*file_info.as_ptr()).attribute & !FILE_PERMISSIONS_MASK) | perm.to_attr()
    };

    f.set_file_info(file_info)
}

pub fn set_times(p: &Path, times: FileTimes) -> io::Result<()> {
    // UEFI does not support symlinks
    set_times_nofollow(p, times)
}

pub fn set_times_nofollow(p: &Path, times: FileTimes) -> io::Result<()> {
    let f = uefi_fs::File::from_path(p, file::MODE_READ | file::MODE_WRITE, 0)?;
    let mut file_info = f.file_info()?;

    if let Some(x) = times.accessed {
        unsafe {
            (*file_info.as_mut_ptr()).last_access_time = uefi_fs::systemtime_to_uefi(x);
        }
    }

    if let Some(x) = times.modified {
        unsafe {
            (*file_info.as_mut_ptr()).modification_time = uefi_fs::systemtime_to_uefi(x);
        }
    }

    f.set_file_info(file_info)
}

pub fn rmdir(p: &Path) -> io::Result<()> {
    let f = uefi_fs::File::from_path(p, file::MODE_READ | file::MODE_WRITE, 0)?;
    let file_info = f.file_info()?;
    let file_attr = FileAttr::from_uefi(file_info);

    if file_attr.file_type().is_dir() {
        f.delete()
    } else {
        Err(io::const_error!(io::ErrorKind::NotADirectory, "expected a directory but got a file"))
    }
}

pub fn remove_dir_all(_path: &Path) -> io::Result<()> {
    unsupported()
}

pub fn exists(path: &Path) -> io::Result<bool> {
    let f = uefi_fs::File::from_path(path, r_efi::protocols::file::MODE_READ, 0);
    match f {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
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
    let f = uefi_fs::File::from_path(p, r_efi::protocols::file::MODE_READ, 0)?;
    let inf = f.file_info()?;
    Ok(FileAttr::from_uefi(inf))
}

pub fn lstat(p: &Path) -> io::Result<FileAttr> {
    stat(p)
}

pub fn canonicalize(p: &Path) -> io::Result<PathBuf> {
    crate::path::absolute(p)
}

pub fn copy(_from: &Path, _to: &Path) -> io::Result<u64> {
    unsupported()
}

mod uefi_fs {
    use r_efi::protocols::{device_path, file, simple_file_system};

    use crate::boxed::Box;
    use crate::ffi::OsString;
    use crate::io;
    use crate::os::uefi::ffi::OsStringExt;
    use crate::path::Path;
    use crate::ptr::NonNull;
    use crate::sys::pal::helpers::{self, UefiBox};
    use crate::sys::time::{self, SystemTime};

    pub(crate) struct File {
        protocol: NonNull<file::Protocol>,
        path: crate::path::PathBuf,
    }

    impl File {
        pub(crate) fn from_path(path: &Path, open_mode: u64, attr: u64) -> io::Result<Self> {
            let absolute = crate::path::absolute(path)?;

            let p = helpers::OwnedDevicePath::from_text(absolute.as_os_str())?;
            let (vol, mut path_remaining) = Self::open_volume_from_device_path(p.borrow())?;

            let protocol = Self::open(vol, &mut path_remaining, open_mode, attr)?;
            Ok(Self { protocol, path: absolute })
        }

        /// Open Filesystem volume given a devicepath to the volume, or a file/directory in the
        /// volume. The path provided should be absolute UEFI device path, without any UEFI shell
        /// mappings.
        ///
        /// Returns
        /// 1. The volume as a UEFI File
        /// 2. Path relative to the volume.
        ///
        /// For example, given "PciRoot(0x0)/Pci(0x1,0x1)/Ata(Secondary,Slave,0x0)/\abc\run.efi",
        /// this will open the volume "PciRoot(0x0)/Pci(0x1,0x1)/Ata(Secondary,Slave,0x0)"
        /// and return the remaining file path "\abc\run.efi".
        fn open_volume_from_device_path(
            path: helpers::BorrowedDevicePath<'_>,
        ) -> io::Result<(NonNull<file::Protocol>, Box<[u16]>)> {
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

            Err(io::const_error!(io::ErrorKind::NotFound, "Volume Not Found"))
        }

        // Open volume on device_handle using SIMPLE_FILE_SYSTEM_PROTOCOL
        fn open_volume(
            device_handle: NonNull<crate::ffi::c_void>,
        ) -> io::Result<NonNull<file::Protocol>> {
            let simple_file_system_protocol = helpers::open_protocol::<simple_file_system::Protocol>(
                device_handle,
                simple_file_system::PROTOCOL_GUID,
            )?;

            let mut file_protocol = crate::ptr::null_mut();
            let r = unsafe {
                ((*simple_file_system_protocol.as_ptr()).open_volume)(
                    simple_file_system_protocol.as_ptr(),
                    &mut file_protocol,
                )
            };
            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            // Since no error was returned, file protocol should be non-NULL.
            let p = NonNull::new(file_protocol).unwrap();
            Ok(p)
        }

        fn open(
            protocol: NonNull<file::Protocol>,
            path: &mut [u16],
            open_mode: u64,
            attr: u64,
        ) -> io::Result<NonNull<file::Protocol>> {
            let file_ptr = protocol.as_ptr();
            let mut file_opened = crate::ptr::null_mut();

            let r = unsafe {
                ((*file_ptr).open)(file_ptr, &mut file_opened, path.as_mut_ptr(), open_mode, attr)
            };

            if r.is_error() {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            // Since no error was returned, file protocol should be non-NULL.
            let p = NonNull::new(file_opened).unwrap();
            Ok(p)
        }

        pub(crate) fn read_dir_entry(&self) -> io::Result<Option<UefiBox<file::Info>>> {
            let file_ptr = self.protocol.as_ptr();
            let mut buf_size = 0;

            let r = unsafe { ((*file_ptr).read)(file_ptr, &mut buf_size, crate::ptr::null_mut()) };

            if buf_size == 0 {
                return Ok(None);
            }

            assert!(r.is_error());
            if r != r_efi::efi::Status::BUFFER_TOO_SMALL {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            let mut info: UefiBox<file::Info> = UefiBox::new(buf_size)?;
            let r =
                unsafe { ((*file_ptr).read)(file_ptr, &mut buf_size, info.as_mut_ptr().cast()) };

            if r.is_error() {
                Err(io::Error::from_raw_os_error(r.as_usize()))
            } else {
                Ok(Some(info))
            }
        }

        pub(crate) fn file_info(&self) -> io::Result<UefiBox<file::Info>> {
            let file_ptr = self.protocol.as_ptr();
            let mut info_id = file::INFO_ID;
            let mut buf_size = 0;

            let r = unsafe {
                ((*file_ptr).get_info)(
                    file_ptr,
                    &mut info_id,
                    &mut buf_size,
                    crate::ptr::null_mut(),
                )
            };
            assert!(r.is_error());
            if r != r_efi::efi::Status::BUFFER_TOO_SMALL {
                return Err(io::Error::from_raw_os_error(r.as_usize()));
            }

            let mut info: UefiBox<file::Info> = UefiBox::new(buf_size)?;
            let r = unsafe {
                ((*file_ptr).get_info)(
                    file_ptr,
                    &mut info_id,
                    &mut buf_size,
                    info.as_mut_ptr().cast(),
                )
            };

            if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(info) }
        }

        pub(crate) fn set_file_info(&self, mut info: UefiBox<file::Info>) -> io::Result<()> {
            let file_ptr = self.protocol.as_ptr();
            let mut info_id = file::INFO_ID;

            let r = unsafe {
                ((*file_ptr).set_info)(file_ptr, &mut info_id, info.len(), info.as_mut_ptr().cast())
            };

            if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
        }

        pub(crate) fn delete(self) -> io::Result<()> {
            let file_ptr = self.protocol.as_ptr();
            let r = unsafe { ((*file_ptr).delete)(file_ptr) };

            // Spec states that even in case of failure, the file handle will be closed.
            crate::mem::forget(self);

            if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
        }

        pub(crate) fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for File {
        fn drop(&mut self) {
            let file_ptr = self.protocol.as_ptr();
            let _ = unsafe { ((*file_ptr).close)(file_ptr) };
        }
    }

    /// A helper to check that target path is a descendent of source. It is expected to be used with
    /// absolute UEFI device paths without any UEFI shell mappings.
    ///
    /// Returns the path relative to source
    ///
    /// For example, given "PciRoot(0x0)/Pci(0x1,0x1)/Ata(Secondary,Slave,0x0)/" and
    /// "PciRoot(0x0)/Pci(0x1,0x1)/Ata(Secondary,Slave,0x0)/\abc\run.efi", this will return
    /// "\abc\run.efi"
    fn path_best_match(
        source: &helpers::BorrowedDevicePath<'_>,
        target: &helpers::BorrowedDevicePath<'_>,
    ) -> Option<Box<[u16]>> {
        let mut source_iter = source.iter().take_while(|x| !x.is_end_instance());
        let mut target_iter = target.iter().take_while(|x| !x.is_end_instance());

        loop {
            match (source_iter.next(), target_iter.next()) {
                (Some(x), Some(y)) if x == y => continue,
                (None, Some(y)) => {
                    let p = y.to_path().to_text().ok()?;
                    return helpers::os_string_to_raw(&p);
                }
                _ => return None,
            }
        }
    }

    /// An implementation of mkdir to allow creating new directory without having to open the
    /// volume twice (once for checking and once for creating)
    pub(crate) fn mkdir(path: &Path) -> io::Result<()> {
        let absolute = crate::path::absolute(path)?;

        let p = helpers::OwnedDevicePath::from_text(absolute.as_os_str())?;
        let (vol, mut path_remaining) = File::open_volume_from_device_path(p.borrow())?;

        // Check if file exists
        match File::open(vol, &mut path_remaining, file::MODE_READ, 0) {
            Ok(_) => {
                return Err(io::Error::new(io::ErrorKind::AlreadyExists, "Path already exists"));
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }

        let _ = File::open(
            vol,
            &mut path_remaining,
            file::MODE_READ | file::MODE_WRITE | file::MODE_CREATE,
            file::DIRECTORY,
        )?;

        Ok(())
    }

    /// EDK2 FAT driver uses EFI_UNSPECIFIED_TIMEZONE to represent localtime. So for proper
    /// conversion to SystemTime, we use the current time to get the timezone in such cases.
    pub(crate) fn uefi_to_systemtime(mut time: r_efi::efi::Time) -> Option<SystemTime> {
        time.timezone = if time.timezone == r_efi::efi::UNSPECIFIED_TIMEZONE {
            time::system_time_internal::now().timezone
        } else {
            time.timezone
        };
        SystemTime::from_uefi(time)
    }

    /// Convert to UEFI Time with the current timezone.
    pub(crate) fn systemtime_to_uefi(time: SystemTime) -> r_efi::efi::Time {
        let now = time::system_time_internal::now();
        time.to_uefi_loose(now.timezone, now.daylight)
    }

    pub(crate) fn file_name_from_uefi(info: &UefiBox<file::Info>) -> OsString {
        let file_name = {
            let size = unsafe { (*info.as_ptr()).size };
            let strlen = (size as usize - crate::mem::size_of::<file::Info<0>>() - 1) / 2;
            unsafe { crate::slice::from_raw_parts((*info.as_ptr()).file_name.as_ptr(), strlen) }
        };

        OsString::from_wide(file_name)
    }
}
