//! Implementation of path from UEFI. Mostly just copying Windows Implementation

use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};

pub const MAIN_SEP_STR: &str = "\\";
pub const MAIN_SEP: char = '\\';

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'\\'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'\\'
}

/// # Safety
///
/// `bytes` must be a valid UTF-8 encoded slice
#[inline]
unsafe fn bytes_as_os_str(bytes: &[u8]) -> &OsStr {
    // &OsStr is the same as &Slice for UEFI
    unsafe { crate::mem::transmute(bytes) }
}

#[inline]
pub fn parse_prefix(p: &OsStr) -> Option<Prefix<'_>> {
    let pos = p.bytes().iter().take_while(|b| !is_sep_byte(**b)).count();
    if pos == 0 || pos == p.bytes().len() {
        // Relative Path
        None
    } else {
        if p.bytes()[pos - 1] == b'/' {
            let prefix = unsafe { bytes_as_os_str(&p.bytes()[0..pos]) };
            Some(Prefix::UefiDevice(prefix))
        } else {
            // The between UEFI prefix and file-path seems to be `/\`
            None
        }
    }
}

pub(crate) fn absolute(path: &Path) -> io::Result<PathBuf> {
    match parse_prefix(path.as_os_str()) {
        // If no prefix, then use the current prefix
        None => match crate::env::current_dir() {
            Ok(x) => Ok(x.join(format!("\\{}", path.to_string_lossy()))),
            Err(_) => Err(io::error::const_io_error!(
                io::ErrorKind::Other,
                "Failed to convert to absolute path"
            )),
        },
        // If Device Path Prefix present, then path should already be absolute
        Some(_) => Ok(path.to_path_buf()),
    }
}
