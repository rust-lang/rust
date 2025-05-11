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
    #[rustfmt::skip]
    let msg = match r_efi::efi::Status::from_usize(errno) {
        Status::ABORTED => "The operation was aborted.",
        Status::ACCESS_DENIED => "Access was denied.",
        Status::ALREADY_STARTED => "The protocol has already been started.",
        Status::BAD_BUFFER_SIZE => "The buffer was not the proper size for the request.",
        Status::BUFFER_TOO_SMALL => "The buffer is not large enough to hold the requested data. The required buffer size is returned in the appropriate parameter when this error occurs.",
        Status::COMPROMISED_DATA => "The security status of the data is unknown or compromised and the data must be updated or replaced to restore a valid security status.",
        Status::CONNECTION_FIN => "The receiving operation fails because the communication peer has closed the connection and there is no more data in the receive buffer of the instance.",
        Status::CONNECTION_REFUSED => "The receiving or transmission operation fails because this connection is refused.",
        Status::CONNECTION_RESET => "The connect fails because the connection is reset either by instance itself or the communication peer.",
        Status::CRC_ERROR => "A CRC error was detected.",
        Status::DEVICE_ERROR => "The physical device reported an error while attempting the operation.",
        Status::END_OF_FILE => "The end of the file was reached.",
        Status::END_OF_MEDIA => "Beginning or end of media was reached",
        Status::HOST_UNREACHABLE => "The remote host is not reachable.",
        Status::HTTP_ERROR => "A HTTP error occurred during the network operation.",
        Status::ICMP_ERROR => "An ICMP error occurred during the network operation.",
        Status::INCOMPATIBLE_VERSION => "The function encountered an internal version that was incompatible with a version requested by the caller.",
        Status::INVALID_LANGUAGE => "The language specified was invalid.",
        Status::INVALID_PARAMETER => "A parameter was incorrect.",
        Status::IP_ADDRESS_CONFLICT => "There is an address conflict address allocation",
        Status::LOAD_ERROR => "The image failed to load.",
        Status::MEDIA_CHANGED => "The medium in the device has changed since the last access.",
        Status::NETWORK_UNREACHABLE => "The network containing the remote host is not reachable.",
        Status::NO_MAPPING => "A mapping to a device does not exist.",
        Status::NO_MEDIA => "The device does not contain any medium to perform the operation.",
        Status::NO_RESPONSE => "The server was not found or did not respond to the request.",
        Status::NOT_FOUND => "The item was not found.",
        Status::NOT_READY => "There is no data pending upon return.",
        Status::NOT_STARTED => "The protocol has not been started.",
        Status::OUT_OF_RESOURCES => "A resource has run out.",
        Status::PROTOCOL_ERROR => "A protocol error occurred during the network operation.",
        Status::PROTOCOL_UNREACHABLE => "An ICMP protocol unreachable error is received.",
        Status::SECURITY_VIOLATION => "The function was not performed due to a security violation.",
        Status::TFTP_ERROR => "A TFTP error occurred during the network operation.",
        Status::TIMEOUT => "The timeout time expired.",
        Status::UNSUPPORTED => "The operation is not supported.",
        Status::VOLUME_FULL => "There is no more space on the file system.",
        Status::VOLUME_CORRUPTED => "An inconstancy was detected on the file system causing the operating to fail.",
        Status::WRITE_PROTECTED => "The device cannot be written to.",
        _ => return format!("Status: {errno}"),
    };
    msg.to_owned()
}

pub fn getcwd() -> io::Result<PathBuf> {
    match helpers::open_shell() {
        Some(shell) => {
            // SAFETY: path_ptr is managed by UEFI shell and should not be deallocated
            let path_ptr = unsafe { ((*shell.as_ptr()).get_cur_dir)(crate::ptr::null_mut()) };
            helpers::os_string_from_raw(path_ptr)
                .map(PathBuf::from)
                .ok_or(io::const_error!(io::ErrorKind::InvalidData, "invalid path"))
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
        .ok_or(io::const_error!(io::ErrorKind::InvalidData, "invalid path"))?;

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
