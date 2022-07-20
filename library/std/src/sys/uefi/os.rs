use super::unsupported;
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::os::uefi;
use crate::path::{self, PathBuf};

pub fn errno() -> i32 {
    uefi::raw::Status::ABORTED.as_usize() as i32
}

pub fn error_string(_errno: i32) -> String {
    "ABORTED".to_string()
}

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
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

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
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

// FIXME: Use GetVariable() method
pub fn getenv(key: &OsStr) -> Option<OsString> {
    uefi_vars::get_variable(key)
}

pub fn setenv(key: &OsStr, val: &OsStr) -> io::Result<()> {
    uefi_vars::set_variable(key, val)
}

pub fn unsetenv(key: &OsStr) -> io::Result<()> {
    uefi_vars::set_variable(key, OsStr::new(""))
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on this platform")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    let code = match usize::try_from(code) {
        Ok(x) => uefi::raw::Status::from_usize(x),
        Err(_) => uefi::raw::Status::ABORTED,
    };

    if let (Some(boot_services), Some(handle)) =
        (uefi::env::get_boot_services(), uefi::env::get_system_handle())
    {
        let _ =
            unsafe { ((*boot_services.as_ptr()).exit)(handle.as_ptr(), code, 0, [0].as_mut_ptr()) };
    }

    crate::intrinsics::abort()
}

pub fn getpid() -> u32 {
    panic!("no pids on this platform")
}

mod uefi_vars {
    // It is possible to directly store and use UTF-8 data. So no need to convert to and from UCS-2
    use super::super::common;
    use crate::ffi::{OsStr, OsString};
    use crate::io;
    use crate::os::uefi;
    use crate::os::uefi::ffi::{OsStrExt, OsStringExt};

    const ENVIRONMENT_GUID: uefi::raw::Guid = uefi::raw::Guid::from_fields(
        0x49bb4029,
        0x7d2b,
        0x4bf7,
        0xa1,
        0x95,
        &[0x0f, 0x18, 0xa1, 0xa8, 0x85, 0xc9],
    );

    pub fn set_variable(key: &OsStr, val: &OsStr) -> io::Result<()> {
        set_variable_inner(key, val.bytes(), uefi::raw::VARIABLE_BOOTSERVICE_ACCESS)
    }

    pub fn append_variable(key: &OsStr, val: &[u8]) -> io::Result<()> {
        set_variable_inner(
            key,
            val,
            uefi::raw::VARIABLE_BOOTSERVICE_ACCESS | uefi::raw::VARIABLE_APPEND_WRITE,
        )
    }

    fn set_variable_inner(key: &OsStr, val: &[u8], attr: u32) -> io::Result<()> {
        let runtime_services = uefi::env::get_runtime_services().ok_or(io::Error::new(
            io::ErrorKind::Uncategorized,
            "Failed to Acquire Runtime Services",
        ))?;
        // Store a copy of data since it is technically possible to manipulate it from other
        // applications
        let mut val_copy = val.to_vec();
        let val_len = val_copy.len();
        let r = unsafe {
            ((*runtime_services.as_ptr()).set_variable)(
                key.to_ffi_string().as_mut_ptr(),
                &mut ENVIRONMENT_GUID,
                attr,
                val_len,
                val_copy.as_mut_ptr().cast(),
            )
        };

        if r.is_error() { Err(common::status_to_io_error(&r)) } else { Ok(()) }
    }

    pub fn get_variable(key: &OsStr) -> Option<OsString> {
        let runtime_services = uefi::env::get_runtime_services()?;
        let mut buf_size = 0;
        let r = unsafe {
            ((*runtime_services.as_ptr()).get_variable)(
                key.to_ffi_string().as_mut_ptr(),
                &mut ENVIRONMENT_GUID,
                crate::ptr::null_mut(),
                &mut buf_size,
                crate::ptr::null_mut(),
            )
        };

        if r.is_error() && r != uefi::raw::Status::BUFFER_TOO_SMALL {
            return None;
        }

        let mut buf: Vec<u8> = Vec::with_capacity(buf_size);
        let r = unsafe {
            ((*runtime_services.as_ptr()).get_variable)(
                key.to_ffi_string().as_mut_ptr(),
                &mut ENVIRONMENT_GUID,
                crate::ptr::null_mut(),
                &mut buf_size,
                buf.as_mut_ptr().cast(),
            )
        };

        if r.is_error() {
            None
        } else {
            unsafe { buf.set_len(buf_size) };
            Some(OsString::from(String::from_utf8(buf).ok()?))
        }
    }
}
