use crate::alloc::{Layout, alloc, dealloc};
use crate::ffi::c_void;
use crate::mem::offset_of;
use crate::os::windows::io::{
    AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, HandleOrInvalid, IntoRawHandle,
    OwnedHandle, RawHandle,
};
use crate::path::Path;
use crate::sys::api::{self, SetFileInformation, UnicodeStrRef, WinError};
use crate::sys::fs::windows::debug_path_handle;
use crate::sys::fs::{File, OpenOptions};
use crate::sys::handle::Handle;
use crate::sys::path::{WCStr, with_native_path};
use crate::sys::{AsInner, FromInner, IntoInner, IoResult, c, to_u16s};
use crate::{fmt, fs, io, ptr};

pub struct Dir {
    handle: Handle,
}

fn run_path_with_u16s<T>(path: &Path, f: &dyn Fn(&[u16]) -> io::Result<T>) -> io::Result<T> {
    let path = to_u16s(path)?;
    f(&path[..path.len() - 1])
}

/// A wrapper around a raw NtCreateFile call.
///
/// This isn't completely safe because `OBJECT_ATTRIBUTES` contains raw pointers.
unsafe fn nt_create_file(
    opts: &OpenOptions,
    object_attributes: &c::OBJECT_ATTRIBUTES,
    create_options: c::NTCREATEFILE_CREATE_OPTIONS,
) -> io::Result<Handle> {
    let mut handle = ptr::null_mut();
    let mut io_status = c::IO_STATUS_BLOCK::PENDING;
    // SYNCHRONIZE is included in FILE_GENERIC_READ, but not GENERIC_READ, so we add it manually
    let access = opts.get_access_mode()? | c::SYNCHRONIZE;
    // one of FILE_SYNCHRONOUS_IO_{,NON}ALERT is required for later operations to succeed.
    let options = create_options | c::FILE_SYNCHRONOUS_IO_NONALERT;
    let status = unsafe {
        c::NtCreateFile(
            &mut handle,
            access,
            object_attributes,
            &mut io_status,
            ptr::null(),
            c::FILE_ATTRIBUTE_NORMAL,
            opts.share_mode,
            opts.get_disposition()?,
            options,
            ptr::null(),
            0,
        )
    };
    if c::nt_success(status) {
        // SAFETY: nt_success guarantees that handle is no longer null
        unsafe { Ok(Handle::from_raw_handle(handle)) }
    } else {
        Err(WinError::new(unsafe { c::RtlNtStatusToDosError(status) })).io_result()
    }
}

impl Dir {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<Self> {
        with_native_path(path, &|path| Self::open_with_native(path, opts))
    }

    pub fn open_file(&self, path: &Path, opts: &OpenOptions) -> io::Result<File> {
        // NtCreateFile will fail if given an absolute path and a non-null RootDirectory
        if path.is_absolute() {
            return File::open(path, opts);
        }
        run_path_with_u16s(path, &|path| {
            self.open_file_native(&path, opts, false).map(|handle| File { handle })
        })
    }

    pub fn remove_file(&self, path: &Path) -> io::Result<()> {
        run_path_with_u16s(path, &|path| self.remove_native(path, false))
    }

    pub fn rename(&self, from: &Path, to_dir: &Self, to: &Path) -> io::Result<()> {
        let is_dir = from.is_dir();
        run_path_with_u16s(from, &|from| {
            run_path_with_u16s(to, &|to| self.rename_native(from, to_dir, to, is_dir))
        })
    }

    fn open_with_native(path: &WCStr, opts: &OpenOptions) -> io::Result<Self> {
        let creation = opts.get_creation_mode()?;
        let sa = c::SECURITY_ATTRIBUTES {
            nLength: size_of::<c::SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: ptr::null_mut(),
            bInheritHandle: opts.inherit_handle as c::BOOL,
        };
        let handle = unsafe {
            c::CreateFileW(
                path.as_ptr(),
                opts.get_access_mode()?,
                opts.share_mode,
                &raw const sa,
                creation,
                // FILE_FLAG_BACKUP_SEMANTICS is required to open a directory
                opts.get_flags_and_attributes() | c::FILE_FLAG_BACKUP_SEMANTICS,
                ptr::null_mut(),
            )
        };
        match OwnedHandle::try_from(unsafe { HandleOrInvalid::from_raw_handle(handle) }) {
            Ok(handle) => Ok(Self { handle: Handle::from_inner(handle) }),
            Err(_) => Err(io::Error::last_os_error()),
        }
    }

    fn open_file_native(&self, path: &[u16], opts: &OpenOptions, dir: bool) -> io::Result<Handle> {
        let name = UnicodeStrRef::from_slice(path);
        let object_attributes = c::OBJECT_ATTRIBUTES {
            RootDirectory: self.handle.as_raw_handle(),
            ObjectName: name.as_ptr(),
            ..c::OBJECT_ATTRIBUTES::with_length()
        };
        let create_opt = if dir { c::FILE_DIRECTORY_FILE } else { c::FILE_NON_DIRECTORY_FILE };
        unsafe { nt_create_file(opts, &object_attributes, create_opt) }
    }

    fn remove_native(&self, path: &[u16], dir: bool) -> io::Result<()> {
        let mut opts = OpenOptions::new();
        opts.access_mode(c::DELETE);
        let handle = self.open_file_native(path, &opts, dir)?;
        let info = c::FILE_DISPOSITION_INFO_EX { Flags: c::FILE_DISPOSITION_FLAG_DELETE };
        let result = unsafe {
            c::SetFileInformationByHandle(
                handle.as_raw_handle(),
                c::FileDispositionInfoEx,
                (&info).as_ptr(),
                size_of::<c::FILE_DISPOSITION_INFO_EX>() as _,
            )
        };
        if result == 0 { Err(api::get_last_error()).io_result() } else { Ok(()) }
    }

    fn rename_native(&self, from: &[u16], to_dir: &Self, to: &[u16], dir: bool) -> io::Result<()> {
        let mut opts = OpenOptions::new();
        opts.access_mode(c::DELETE);
        opts.custom_flags(c::FILE_FLAG_OPEN_REPARSE_POINT | c::FILE_FLAG_BACKUP_SEMANTICS);
        let handle = self.open_file_native(from, &opts, dir)?;
        // Calculate the layout of the `FILE_RENAME_INFORMATION` we pass to `NtSetInformationFile`
        // This is a dynamically sized struct so we need to get the position of the last field to calculate the actual size.
        const too_long_err: io::Error =
            io::const_error!(io::ErrorKind::InvalidFilename, "Filename too long");
        let struct_size = to
            .len()
            .checked_mul(2)
            .and_then(|x| x.checked_add(offset_of!(c::FILE_RENAME_INFORMATION, FileName)))
            .ok_or(too_long_err)?;
        let layout = Layout::from_size_align(struct_size, align_of::<c::FILE_RENAME_INFORMATION>())
            .map_err(|_| too_long_err)?;
        let struct_size = u32::try_from(struct_size).map_err(|_| too_long_err)?;
        let to_byte_len = u32::try_from(to.len() * 2).map_err(|_| too_long_err)?;

        let file_rename_info;
        // SAFETY: We allocate enough memory for a full FILE_RENAME_INFORMATION struct and the filename.
        unsafe {
            file_rename_info = alloc(layout).cast::<c::FILE_RENAME_INFORMATION>();
            if file_rename_info.is_null() {
                return Err(io::ErrorKind::OutOfMemory.into());
            }

            (&raw mut (*file_rename_info).Anonymous).write(c::FILE_RENAME_INFORMATION_0 {
                Flags: c::FILE_RENAME_FLAG_REPLACE_IF_EXISTS | c::FILE_RENAME_FLAG_POSIX_SEMANTICS,
            });

            (&raw mut (*file_rename_info).RootDirectory).write(to_dir.handle.as_raw_handle());
            // Don't include the NULL in the size
            (&raw mut (*file_rename_info).FileNameLength).write(to_byte_len);

            to.as_ptr().copy_to_nonoverlapping(
                (&raw mut (*file_rename_info).FileName).cast::<u16>(),
                to.len(),
            );
        }

        let status = unsafe {
            c::NtSetInformationFile(
                handle.as_raw_handle(),
                &mut c::IO_STATUS_BLOCK::default(),
                file_rename_info.cast::<c_void>(),
                struct_size,
                c::FileRenameInformation,
            )
        };
        unsafe { dealloc(file_rename_info.cast::<u8>(), layout) };
        if c::nt_success(status) {
            // SAFETY: nt_success guarantees that handle is no longer null
            Ok(())
        } else {
            Err(WinError::new(unsafe { c::RtlNtStatusToDosError(status) }))
        }
        .io_result()
    }
}

impl fmt::Debug for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b = debug_path_handle(self.handle.as_handle(), f, "Dir");
        b.finish()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl AsRawHandle for fs::Dir {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle.as_raw_handle()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl IntoRawHandle for fs::Dir {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().handle.into_raw_handle()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl FromRawHandle for fs::Dir {
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        Self::from_inner(Dir { handle: unsafe { FromRawHandle::from_raw_handle(handle) } })
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl AsHandle for fs::Dir {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.as_inner().handle.as_handle()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl From<fs::Dir> for OwnedHandle {
    fn from(value: fs::Dir) -> Self {
        value.into_inner().handle.into_inner()
    }
}

#[unstable(feature = "dirfd", issue = "120426")]
impl From<OwnedHandle> for fs::Dir {
    fn from(value: OwnedHandle) -> Self {
        Self::from_inner(Dir { handle: Handle::from_inner(value) })
    }
}
