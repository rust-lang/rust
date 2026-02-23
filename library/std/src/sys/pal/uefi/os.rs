use r_efi::efi::Status;
use r_efi::efi::protocols::{device_path, loaded_image_device_path};

use super::{helpers, unsupported_err};
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::uefi;
use crate::os::uefi::ffi::{OsStrExt, OsStringExt};
use crate::path::{self, PathBuf};
use crate::ptr::NonNull;
use crate::{fmt, io};

const PATHS_SEP: u16 = b';' as u16;

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

// UEFI Shell Path variable is defined in Section 3.6.1
// [UEFI Shell Specification](https://uefi.org/sites/default/files/resources/UEFI_Shell_2_2.pdf).
pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    let mut joined = Vec::new();

    for (i, path) in paths.enumerate() {
        if i > 0 {
            joined.push(PATHS_SEP)
        }

        let v = path.as_ref().encode_wide().collect::<Vec<u16>>();
        if v.contains(&PATHS_SEP) {
            return Err(JoinPathsError);
        }

        joined.extend_from_slice(&v);
    }

    Ok(OsString::from_wide(&joined))
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "path segment contains `;`".fmt(f)
    }
}

impl crate::error::Error for JoinPathsError {}

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
