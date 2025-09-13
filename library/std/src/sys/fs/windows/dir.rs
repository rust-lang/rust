use crate::os::windows::io::{
    AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, HandleOrInvalid, IntoRawHandle,
    OwnedHandle, RawHandle,
};
use crate::path::Path;
use crate::sys::api::{UnicodeStrRef, WinError};
use crate::sys::fs::windows::debug_path_handle;
use crate::sys::fs::{File, OpenOptions};
use crate::sys::handle::Handle;
use crate::sys::path::{WCStr, with_native_path, with_native_path_bytes};
use crate::sys::{IoResult, c};
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::{fmt, fs, io, ptr};

pub struct Dir {
    handle: Handle,
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
    let access = opts.get_access_mode()?;
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
        if path.is_absolute() {
            return File::open(path, opts);
        }
        with_native_path_bytes(path, &|path| self.open_file_native(path, opts))
            .map(|handle| File { handle })
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
                opts.get_flags_and_attributes() | c::FILE_FLAG_BACKUP_SEMANTICS,
                ptr::null_mut(),
            )
        };
        match OwnedHandle::try_from(unsafe { HandleOrInvalid::from_raw_handle(handle) }) {
            Ok(handle) => Ok(Self { handle: Handle::from_inner(handle) }),
            Err(_) => Err(io::Error::last_os_error()),
        }
    }

    fn open_file_native(&self, path: &[u16], opts: &OpenOptions) -> io::Result<Handle> {
        let name = UnicodeStrRef::from_slice(path);
        let object_attributes = c::OBJECT_ATTRIBUTES {
            RootDirectory: self.handle.as_raw_handle(),
            ObjectName: name.as_ptr(),
            ..c::OBJECT_ATTRIBUTES::with_length()
        };
        unsafe { nt_create_file(opts, &object_attributes, c::FILE_DIRECTORY_FILE) }
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

#[unstable(feature = "dirhandle", issue = "120426")]
impl IntoRawHandle for fs::Dir {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().handle.into_raw_handle()
    }
}

#[unstable(feature = "dirhandle", issue = "120426")]
impl FromRawHandle for fs::Dir {
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        Self::from_inner(Dir { handle: unsafe { FromRawHandle::from_raw_handle(handle) } })
    }
}

#[unstable(feature = "dirhandle", issue = "120426")]
impl AsHandle for fs::Dir {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.as_inner().handle.as_handle()
    }
}

#[unstable(feature = "dirhandle", issue = "120426")]
impl From<fs::Dir> for OwnedHandle {
    fn from(value: fs::Dir) -> Self {
        value.into_inner().handle.into_inner()
    }
}

#[unstable(feature = "dirhandle", issue = "120426")]
impl From<OwnedHandle> for fs::Dir {
    fn from(value: OwnedHandle) -> Self {
        Self::from_inner(Dir { handle: Handle::from_inner(value) })
    }
}
