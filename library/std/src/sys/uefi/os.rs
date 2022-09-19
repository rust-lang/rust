use super::{
    common::{self, status_to_io_error},
    unsupported,
};
use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::os::uefi;
use crate::path::{self, PathBuf};
use crate::{error::Error as StdError, os::uefi::ffi::OsStringExt};

// Return EFI_ABORTED as Status
pub fn errno() -> i32 {
    r_efi::efi::Status::SUCCESS.as_usize() as i32
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

    match common::get_current_handle_protocol::<device_path::Protocol>(
        loaded_image_device_path::PROTOCOL_GUID,
    ) {
        Some(mut x) => unsafe { super::path::device_path_to_path(x.as_mut()) },
        None => Err(io::const_io_error!(
            io::ErrorKind::Uncategorized,
            "failed to acquire EFI_LOADED_IMAGE_DEVICE_PATH_PROTOCOL",
        )),
    }
}

// FIXME: Implement using Variable Services
pub struct Env {
    last_var_name: Vec<u16>,
    last_var_guid: r_efi::efi::Guid,
}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        let mut key = self.last_var_name.clone();
        // Only Read Shell and Rust variables. UEFI variables can be random bytes of data and so
        // not much point in reading anything else in Env context.
        let val = match self.last_var_guid {
            uefi_vars::RUST_VARIABLE_GUID => {
                uefi_vars::get_variable_utf8(&mut key, uefi_vars::RUST_VARIABLE_GUID)
            }
            uefi_vars::SHELL_VARIABLE_GUID => {
                uefi_vars::get_variable_ucs2(&mut key, self.last_var_guid)
            }
            _ => None,
        };
        let (k, g) =
            uefi_vars::get_next_variable_name(&self.last_var_name, &self.last_var_guid).ok()?;
        self.last_var_guid = g;
        self.last_var_name = k;
        match val {
            None => self.next(),
            Some(x) => Some((OsString::from_wide(&key[..(key.len() - 1)]), x)),
        }
    }
}

pub fn env() -> Env {
    // The Guid should be ignored, so just passing anything
    let (key, guid) =
        uefi_vars::get_next_variable_name(&[0], &uefi_vars::RUST_VARIABLE_GUID).unwrap();
    Env { last_var_name: key, last_var_guid: guid }
}

// Tries to get variable from bot RUST_VARIABLE_GUID and SHELL_VARIABLE_GUID. Precedence:
// RUST_VARIABLE_GUID > SHELL_VARIABLE_GUID
pub fn getenv(key: &OsStr) -> Option<OsString> {
    let mut k = common::to_ffi_string(key);
    if let Some(x) = uefi_vars::get_variable_utf8(&mut k, uefi_vars::RUST_VARIABLE_GUID) {
        Some(x)
    } else {
        uefi_vars::get_variable_ucs2(&mut k, uefi_vars::SHELL_VARIABLE_GUID)
    }
}

// Only possible to set variable to RUST_VARIABLE_GUID
pub fn setenv(key: &OsStr, val: &OsStr) -> io::Result<()> {
    // Setting a variable with null value is same as unsetting it in UEFI
    if val.is_empty() {
        unsetenv(key)
    } else {
        unsafe {
            uefi_vars::set_variable_raw(
                common::to_ffi_string(key).as_mut_ptr(),
                uefi_vars::RUST_VARIABLE_GUID,
                val.len(),
                val.bytes().as_ptr() as *mut crate::ffi::c_void,
            )
        }
    }
}

pub fn unsetenv(key: &OsStr) -> io::Result<()> {
    match unsafe {
        uefi_vars::set_variable_raw(
            common::to_ffi_string(key).as_mut_ptr(),
            uefi_vars::RUST_VARIABLE_GUID,
            0,
            crate::ptr::null_mut(),
        )
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

    let handle = uefi::env::image_handle().cast();
    // First try to use EFI_BOOT_SERVICES.Exit()
    if let Some(boot_services) = common::get_boot_services() {
        let _ = unsafe {
            ((*boot_services.as_ptr()).exit)(handle.as_ptr(), code, 0, crate::ptr::null_mut())
        };
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
    use super::super::common::{self, status_to_io_error};
    use crate::ffi::OsString;
    use crate::io;
    use crate::mem::size_of;
    use crate::os::uefi::ffi::OsStringExt;

    // Using Shell Variable Guid from edk2/ShellPkg
    pub(crate) const SHELL_VARIABLE_GUID: r_efi::efi::Guid = r_efi::efi::Guid::from_fields(
        0x158def5a,
        0xf656,
        0x419c,
        0xb0,
        0x27,
        &[0x7a, 0x31, 0x92, 0xc0, 0x79, 0xd2],
    );

    pub(crate) const RUST_VARIABLE_GUID: r_efi::efi::Guid = r_efi::efi::Guid::from_fields(
        0x49bb4029,
        0x7d2b,
        0x4bf7,
        0xa1,
        0x95,
        &[0x0f, 0x18, 0xa1, 0xa8, 0x85, 0xc9],
    );

    pub(crate) fn get_variable_utf8(key: &mut [u16], guid: r_efi::efi::Guid) -> Option<OsString> {
        let mut buf_size = 0;

        if let Err(e) = unsafe {
            get_vaiable_raw(key.as_mut_ptr(), guid, &mut buf_size, crate::ptr::null_mut())
        } {
            if e.kind() != io::ErrorKind::FileTooLarge {
                return None;
            }
        }

        let mut buf: Vec<u8> = Vec::with_capacity(buf_size);
        unsafe {
            get_vaiable_raw(key.as_mut_ptr(), guid, &mut buf_size, buf.as_mut_ptr().cast()).ok()
        }?;

        unsafe { buf.set_len(buf_size) };
        Some(OsString::from(String::from_utf8(buf).ok()?))
    }

    pub(crate) fn get_variable_ucs2(key: &mut [u16], guid: r_efi::efi::Guid) -> Option<OsString> {
        let mut buf_size = 0;

        if let Err(e) = unsafe {
            get_vaiable_raw(key.as_mut_ptr(), guid, &mut buf_size, crate::ptr::null_mut())
        } {
            if e.kind() != io::ErrorKind::FileTooLarge {
                return None;
            }
        }

        let mut buf: Vec<u16> = Vec::with_capacity(buf_size / size_of::<u16>());
        unsafe {
            get_vaiable_raw(key.as_mut_ptr(), guid, &mut buf_size, buf.as_mut_ptr().cast()).ok()
        }?;

        unsafe { buf.set_len(buf_size / size_of::<u16>()) };
        Some(OsString::from_wide(&buf))
    }

    pub(crate) fn get_next_variable_name(
        last_var_name: &[u16],
        last_guid: &r_efi::efi::Guid,
    ) -> io::Result<(Vec<u16>, r_efi::efi::Guid)> {
        #[inline]
        fn buf_size(s: usize) -> usize {
            s / size_of::<u16>()
        }

        let mut var_name = Vec::from(last_var_name);
        let mut var_size = var_name.capacity() * size_of::<u16>();
        let mut guid: r_efi::efi::Guid = *last_guid;
        match unsafe { get_next_variable_raw(&mut var_size, var_name.as_mut_ptr(), &mut guid) } {
            Ok(_) => {
                unsafe { var_name.set_len(buf_size(var_size)) };
                return Ok((var_name, guid));
            }
            Err(e) => {
                if e.kind() != io::ErrorKind::FileTooLarge {
                    return Err(e);
                }
            }
        }

        var_name.reserve(buf_size(var_size) - var_name.capacity() + 1);
        var_size = var_name.capacity() * size_of::<u16>();

        unsafe { get_next_variable_raw(&mut var_size, var_name.as_mut_ptr(), &mut guid) }?;
        unsafe { var_name.set_len(buf_size(var_size)) };
        Ok((var_name, guid))
    }

    pub(crate) unsafe fn set_variable_raw(
        variable_name: *mut u16,
        mut guid: r_efi::efi::Guid,
        data_size: usize,
        data: *mut crate::ffi::c_void,
    ) -> io::Result<()> {
        let runtime_services =
            common::get_runtime_services().ok_or(common::RUNTIME_SERVICES_ERROR)?;
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

    unsafe fn get_vaiable_raw(
        key: *mut u16,
        mut guid: r_efi::efi::Guid,
        data_size: &mut usize,
        data: *mut crate::ffi::c_void,
    ) -> io::Result<()> {
        let runtime_services =
            common::get_runtime_services().ok_or(common::RUNTIME_SERVICES_ERROR)?;
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

    unsafe fn get_next_variable_raw(
        variable_name_size: *mut usize,
        variable_name: *mut u16,
        vendor_guid: *mut r_efi::efi::Guid,
    ) -> io::Result<()> {
        let runtime_services =
            common::get_runtime_services().ok_or(common::RUNTIME_SERVICES_ERROR)?;
        let r = unsafe {
            ((*runtime_services.as_ptr()).get_next_variable_name)(
                variable_name_size,
                variable_name,
                vendor_guid,
            )
        };
        if r.is_error() { Err(status_to_io_error(r)) } else { Ok(()) }
    }
}
