use r_efi::efi::Status;
use r_efi::efi::protocols::{device_path, loaded_image_device_path};

use super::{RawOsError, helpers, unsupported_err};
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::uefi;
use crate::path::{self, PathBuf};
use crate::ptr::NonNull;
use crate::{fmt, io};

pub fn errno() -> RawOsError {
    0
}

pub fn error_string(errno: RawOsError) -> String {
    // Keep the List in Alphabetical Order
    // The Messages are taken from UEFI Specification Appendix D - Status Codes
    match r_efi::efi::Status::from_usize(errno) {
        Status::ABORTED => "The operation was aborted.".to_owned(),
        Status::ACCESS_DENIED => "Access was denied.".to_owned(),
        Status::ALREADY_STARTED => "The protocol has already been started.".to_owned(),
        Status::BAD_BUFFER_SIZE => "The buffer was not the proper size for the request.".to_owned(),
        Status::BUFFER_TOO_SMALL => {
                "The buffer is not large enough to hold the requested data. The required buffer size is returned in the appropriate parameter when this error occurs.".to_owned()
        }
        Status::COMPROMISED_DATA => {
                "The security status of the data is unknown or compromised and the data must be updated or replaced to restore a valid security status.".to_owned()
        }
        Status::CONNECTION_FIN => {
                "The receiving operation fails because the communication peer has closed the connection and there is no more data in the receive buffer of the instance.".to_owned()
        }
        Status::CONNECTION_REFUSED => {
                "The receiving or transmission operation fails because this connection is refused.".to_owned()
        }
        Status::CONNECTION_RESET => {
                "The connect fails because the connection is reset either by instance itself or the communication peer.".to_owned()
        }
        Status::CRC_ERROR => "A CRC error was detected.".to_owned(),
        Status::DEVICE_ERROR =>             "The physical device reported an error while attempting the operation.".to_owned()
        ,
        Status::END_OF_FILE => {
            "The end of the file was reached.".to_owned()
        }
        Status::END_OF_MEDIA => {
            "Beginning or end of media was reached".to_owned()
        }
        Status::HOST_UNREACHABLE => {
            "The remote host is not reachable.".to_owned()
        }
        Status::HTTP_ERROR => {
            "A HTTP error occurred during the network operation.".to_owned()
        }
        Status::ICMP_ERROR => {
                "An ICMP error occurred during the network operation.".to_owned()
        }
        Status::INCOMPATIBLE_VERSION => {
                "The function encountered an internal version that was incompatible with a version requested by the caller.".to_owned()
        }
        Status::INVALID_LANGUAGE => {
            "The language specified was invalid.".to_owned()
        }
        Status::INVALID_PARAMETER => {
            "A parameter was incorrect.".to_owned()
        }
        Status::IP_ADDRESS_CONFLICT => {
            "There is an address conflict address allocation".to_owned()
        }
        Status::LOAD_ERROR => {
            "The image failed to load.".to_owned()
        }
        Status::MEDIA_CHANGED => {
                "The medium in the device has changed since the last access.".to_owned()
        }
        Status::NETWORK_UNREACHABLE => {
                "The network containing the remote host is not reachable.".to_owned()
        }
        Status::NO_MAPPING => {
            "A mapping to a device does not exist.".to_owned()
        }
        Status::NO_MEDIA => {
                "The device does not contain any medium to perform the operation.".to_owned()
        }
        Status::NO_RESPONSE => {
                "The server was not found or did not respond to the request.".to_owned()
        }
        Status::NOT_FOUND => "The item was not found.".to_owned(),
        Status::NOT_READY => {
            "There is no data pending upon return.".to_owned()
        }
        Status::NOT_STARTED => {
            "The protocol has not been started.".to_owned()
        }
        Status::OUT_OF_RESOURCES => {
            "A resource has run out.".to_owned()
        }
        Status::PROTOCOL_ERROR => {
                "A protocol error occurred during the network operation.".to_owned()
        }
        Status::PROTOCOL_UNREACHABLE => {
            "An ICMP protocol unreachable error is received.".to_owned()
        }
        Status::SECURITY_VIOLATION => {
                "The function was not performed due to a security violation.".to_owned()
        }
        Status::TFTP_ERROR => {
            "A TFTP error occurred during the network operation.".to_owned()
        }
        Status::TIMEOUT => "The timeout time expired.".to_owned(),
        Status::UNSUPPORTED => {
            "The operation is not supported.".to_owned()
        }
        Status::VOLUME_FULL => {
            "There is no more space on the file system.".to_owned()
        }
        Status::VOLUME_CORRUPTED => {
                "An inconstancy was detected on the file system causing the operating to fail.".to_owned()
        }
        Status::WRITE_PROTECTED => {
            "The device cannot be written to.".to_owned()
        }
        _ => format!("Status: {}", errno),
    }
}

pub fn getcwd() -> io::Result<PathBuf> {
    match helpers::open_shell() {
        Some(shell) => {
            // SAFETY: path_ptr is managed by UEFI shell and should not be deallocated
            let path_ptr = unsafe { ((*shell.as_ptr()).get_cur_dir)(crate::ptr::null_mut()) };
            helpers::os_string_from_raw(path_ptr)
                .map(PathBuf::from)
                .ok_or(io::const_io_error!(io::ErrorKind::InvalidData, "Invalid path"))
        }
        None => {
            let mut t = current_exe()?;
            // SAFETY: This should never fail since the disk prefix will be present even for root
            // executables
            assert!(t.pop());
            Ok(t)
        }
    }
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    let shell = helpers::open_shell().ok_or(unsupported_err())?;

    let mut p = helpers::os_string_to_raw(p.as_os_str())
        .ok_or(io::const_io_error!(io::ErrorKind::InvalidData, "Invalid path"))?;

    let r = unsafe { ((*shell.as_ptr()).set_cur_dir)(crate::ptr::null_mut(), p.as_mut_ptr()) };
    if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
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

impl StdError for JoinPathsError {}

pub fn current_exe() -> io::Result<PathBuf> {
    let protocol = helpers::image_handle_protocol::<device_path::Protocol>(
        loaded_image_device_path::PROTOCOL_GUID,
    )?;
    helpers::device_path_to_text(protocol).map(PathBuf::from)
}

pub struct EnvStrDebug<'a> {
    iter: &'a [(OsString, OsString)],
}

impl fmt::Debug for EnvStrDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for (a, b) in self.iter {
            list.entry(&(a.to_str().unwrap(), b.to_str().unwrap()));
        }
        list.finish()
    }
}

pub struct Env(crate::vec::IntoIter<(OsString, OsString)>);

impl Env {
    // FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
    pub fn str_debug(&self) -> impl fmt::Debug + '_ {
        EnvStrDebug { iter: self.0.as_slice() }
    }
}

impl Iterator for Env {
    type Item = (OsString, OsString);

    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.0.next()
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub fn env() -> Env {
    let env = uefi_env::get_all().expect("not supported on this platform");
    Env(env.into_iter())
}

pub fn getenv(key: &OsStr) -> Option<OsString> {
    uefi_env::get(key)
}

pub unsafe fn setenv(key: &OsStr, val: &OsStr) -> io::Result<()> {
    uefi_env::set(key, val)
}

pub unsafe fn unsetenv(key: &OsStr) -> io::Result<()> {
    uefi_env::unset(key)
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on this platform")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    if let (Some(boot_services), Some(handle)) =
        (uefi::env::boot_services(), uefi::env::try_image_handle())
    {
        let boot_services: NonNull<r_efi::efi::BootServices> = boot_services.cast();
        let _ = unsafe {
            ((*boot_services.as_ptr()).exit)(
                handle.as_ptr(),
                Status::from_usize(code as usize),
                0,
                crate::ptr::null_mut(),
            )
        };
    }
    crate::intrinsics::abort()
}

pub fn getpid() -> u32 {
    panic!("no pids on this platform")
}

mod uefi_env {
    use crate::ffi::{OsStr, OsString};
    use crate::io;
    use crate::os::uefi::ffi::OsStringExt;
    use crate::ptr::NonNull;
    use crate::sys::{helpers, unsupported_err};

    pub(crate) fn get(key: &OsStr) -> Option<OsString> {
        let shell = helpers::open_shell()?;
        let mut key_ptr = helpers::os_string_to_raw(key)?;
        unsafe { get_raw(shell, key_ptr.as_mut_ptr()) }
    }

    pub(crate) fn set(key: &OsStr, val: &OsStr) -> io::Result<()> {
        let mut key_ptr = helpers::os_string_to_raw(key)
            .ok_or(io::const_io_error!(io::ErrorKind::InvalidInput, "Invalid Key"))?;
        let mut val_ptr = helpers::os_string_to_raw(val)
            .ok_or(io::const_io_error!(io::ErrorKind::InvalidInput, "Invalid Value"))?;
        unsafe { set_raw(key_ptr.as_mut_ptr(), val_ptr.as_mut_ptr()) }
    }

    pub(crate) fn unset(key: &OsStr) -> io::Result<()> {
        let mut key_ptr = helpers::os_string_to_raw(key)
            .ok_or(io::const_io_error!(io::ErrorKind::InvalidInput, "Invalid Key"))?;
        unsafe { set_raw(key_ptr.as_mut_ptr(), crate::ptr::null_mut()) }
    }

    pub(crate) fn get_all() -> io::Result<Vec<(OsString, OsString)>> {
        let shell = helpers::open_shell().ok_or(unsupported_err())?;

        let mut vars = Vec::new();
        let val = unsafe { ((*shell.as_ptr()).get_env)(crate::ptr::null_mut()) };

        if val.is_null() {
            return Ok(vars);
        }

        let mut start = 0;

        // UEFI Shell returns all keys seperated by NULL.
        // End of string is denoted by two NULLs
        for i in 0.. {
            if unsafe { *val.add(i) } == 0 {
                // Two NULL signal end of string
                if i == start {
                    break;
                }

                let key = OsString::from_wide(unsafe {
                    crate::slice::from_raw_parts(val.add(start), i - start)
                });
                // SAFETY: val.add(start) is always NULL terminated
                let val = unsafe { get_raw(shell, val.add(start)) }
                    .ok_or(io::const_io_error!(io::ErrorKind::InvalidInput, "Invalid Value"))?;

                vars.push((key, val));
                start = i + 1;
            }
        }

        Ok(vars)
    }

    unsafe fn get_raw(
        shell: NonNull<r_efi::efi::protocols::shell::Protocol>,
        key_ptr: *mut r_efi::efi::Char16,
    ) -> Option<OsString> {
        let val = unsafe { ((*shell.as_ptr()).get_env)(key_ptr) };
        helpers::os_string_from_raw(val)
    }

    unsafe fn set_raw(
        key_ptr: *mut r_efi::efi::Char16,
        val_ptr: *mut r_efi::efi::Char16,
    ) -> io::Result<()> {
        let shell = helpers::open_shell().ok_or(unsupported_err())?;
        let r =
            unsafe { ((*shell.as_ptr()).set_env)(key_ptr, val_ptr, r_efi::efi::Boolean::FALSE) };
        if r.is_error() { Err(io::Error::from_raw_os_error(r.as_usize())) } else { Ok(()) }
    }
}
