use super::{common, unsupported};
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::os::uefi;
use crate::os::uefi::ffi::OsStrExt;
use crate::os::uefi::io::status_to_io_error;
use crate::path::{self, PathBuf};

// Return EFI_ABORTED as Status
pub fn errno() -> i32 {
    r_efi::efi::Status::ABORTED.as_usize() as i32
}

pub fn error_string(errno: i32) -> String {
    let r = r_efi::efi::Status::from_usize(errno as usize);
    status_to_io_error(r).to_string()
}

// Implemented using EFI_LOADED_IMAGE_DEVICE_PATH_PROTOCOL
pub fn getcwd() -> io::Result<PathBuf> {
    let mut p = current_exe()?;
    p.pop();
    Ok(p)
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    unsupported()
}

pub struct SplitPaths<'a>(!, PhantomData<&'a ()>);

pub fn split_paths(_unparsed: &OsStr) -> SplitPaths<'_> {
    panic!("unsupported")
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        self.0
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(_paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    Err(JoinPathsError)
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "not supported on this platform yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "not supported on this platform yet"
    }
}

// Implemented using EFI_LOADED_IMAGE_DEVICE_PATH_PROTOCOL
pub fn current_exe() -> io::Result<PathBuf> {
    use r_efi::efi::protocols::{device_path, loaded_image_device_path};

    let mut protocol_guid = loaded_image_device_path::PROTOCOL_GUID;
    match common::get_current_handle_protocol::<device_path::Protocol>(&mut protocol_guid) {
        Some(mut x) => unsafe { super::path::device_path_to_path(x.as_mut()) },
        None => Err(io::error::const_io_error!(
            io::ErrorKind::Uncategorized,
            "Failed to Acquire EFI_LOADED_IMAGE_DEVICE_PATH_PROTOCOL",
        )),
    }
}

// FIXME: Implement using Variable Services
pub struct Env(!);

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.0
    }
}

pub fn env() -> Env {
    panic!("not supported on this platform")
}

pub fn getenv(key: &OsStr) -> Option<OsString> {
    uefi_vars::get_variable(key)
}

pub fn setenv(key: &OsStr, val: &OsStr) -> io::Result<()> {
    // Setting a variable with null value is same as unsetting it in UEFI
    if val.is_empty() {
        unsetenv(key)
    } else {
        unsafe {
            uefi_vars::set_variable(
                key.to_ffi_string().as_mut_ptr(),
                val.len(),
                val.bytes().as_ptr() as *mut crate::ffi::c_void,
            )
        }
    }
}

pub fn unsetenv(key: &OsStr) -> io::Result<()> {
    match unsafe {
        uefi_vars::set_variable(key.to_ffi_string().as_mut_ptr(), 0, crate::ptr::null_mut())
    } {
        Ok(_) => Ok(()),
        Err(e) => match e.kind() {
            // Its fine if the key does not exist
            io::ErrorKind::NotFound => Ok(()),
            _ => Err(e),
        },
    }
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on this platform")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    let code = match usize::try_from(code) {
        Ok(x) => r_efi::efi::Status::from_usize(x),
        Err(_) => r_efi::efi::Status::ABORTED,
    };

    // First try to use EFI_BOOT_SERVICES.Exit()
    if let (Some(boot_services), Some(handle)) =
        (uefi::env::get_boot_services(), uefi::env::get_system_handle())
    {
        let _ =
            unsafe { ((*boot_services.as_ptr()).exit)(handle.as_ptr(), code, 0, [0].as_mut_ptr()) };
    }

    // If exit is not possible, the call abort
    crate::intrinsics::abort()
}

pub fn getpid() -> u32 {
    panic!("no pids on this platform")
}

// Implement variables using Variable Services in EFI_RUNTIME_SERVICES
pub(crate) mod uefi_vars {
    // It is possible to directly store and use UTF-8 data. So no need to convert to and from UCS-2
    use super::super::common;
    use crate::ffi::{OsStr, OsString};
    use crate::io;
    use crate::os::uefi;
    use crate::os::uefi::ffi::OsStrExt;
    use crate::os::uefi::io::status_to_io_error;

    // Using Shell Variable Guid from edk2/ShellPkg
    const SHELL_VARIABLE_GUID: r_efi::efi::Guid = r_efi::efi::Guid::from_fields(
        0x158def5a,
        0xf656,
        0x419c,
        0xb0,
        0x27,
        &[0x7a, 0x31, 0x92, 0xc0, 0x79, 0xd2],
    );

    pub(crate) unsafe fn set_variable(
        variable_name: *mut u16,
        data_size: usize,
        data: *mut crate::ffi::c_void,
    ) -> io::Result<()> {
        let runtime_services =
            uefi::env::get_runtime_services().ok_or(common::RUNTIME_SERVICES_ERROR)?;
        let mut guid = SHELL_VARIABLE_GUID;
        let r = unsafe {
            ((*runtime_services.as_ptr()).set_variable)(
                variable_name,
                &mut guid,
                r_efi::efi::VARIABLE_BOOTSERVICE_ACCESS,
                data_size,
                data,
            )
        };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }

    pub(crate) fn get_variable(key: &OsStr) -> Option<OsString> {
        let mut buf_size = 0;
        let mut key_buf = key.to_ffi_string();

        if let Err(e) =
            unsafe { get_vaiable_raw(key_buf.as_mut_ptr(), &mut buf_size, crate::ptr::null_mut()) }
        {
            if e.kind() != io::ErrorKind::FileTooLarge {
                return None;
            }
        }

        let mut buf: Vec<u8> = Vec::with_capacity(buf_size);
        unsafe { get_vaiable_raw(key_buf.as_mut_ptr(), &mut buf_size, buf.as_mut_ptr().cast()) }
            .ok()?;

        unsafe { buf.set_len(buf_size) };
        Some(OsString::from(String::from_utf8(buf).ok()?))
    }

    unsafe fn get_vaiable_raw(
        key: *mut u16,
        data_size: &mut usize,
        data: *mut crate::ffi::c_void,
    ) -> io::Result<()> {
        let runtime_services =
            uefi::env::get_runtime_services().ok_or(common::RUNTIME_SERVICES_ERROR)?;
        let mut guid = SHELL_VARIABLE_GUID;
        let r = unsafe {
            ((*runtime_services.as_ptr()).get_variable)(
                key,
                &mut guid,
                crate::ptr::null_mut(),
                data_size,
                data,
            )
        };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }
}
