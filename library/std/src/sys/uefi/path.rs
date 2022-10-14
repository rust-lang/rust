//! Implementation of path from UEFI. Mostly just copying Windows Implementation

use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};
use crate::ptr::NonNull;
use crate::sys::uefi::common;

use r_efi::protocols::{device_path, device_path_from_text, device_path_to_text};

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

pub fn parse_prefix(_p: &OsStr) -> Option<Prefix<'_>> {
    None
}

pub(crate) fn absolute(path: &Path) -> io::Result<PathBuf> {
    match device_prefix(path.as_os_str()) {
        // If no prefix, then use the current prefix
        None => match crate::env::current_dir() {
            Ok(x) => {
                if x.as_os_str().bytes().last() == Some(&b'/') {
                    Ok(PathBuf::from(format!(
                        "{}\\{}",
                        x.to_string_lossy(),
                        path.to_string_lossy()
                    )))
                } else {
                    Ok(x.join(format!("\\{}", path.to_string_lossy())))
                }
            }
            Err(_) => {
                Err(io::const_io_error!(io::ErrorKind::Other, "failed to convert to absolute path"))
            }
        },
        // If Device Path Prefix present, then path should already be absolute
        Some(_) => Ok(path.to_path_buf()),
    }
}

pub(crate) fn device_prefix(p: &OsStr) -> Option<&OsStr> {
    let pos = p.bytes().iter().take_while(|b| !is_sep_byte(**b)).count();
    if pos == 0 || pos == p.bytes().len() {
        // Relative Path
        None
    } else {
        if p.bytes()[pos - 1] == b'/' {
            let prefix = unsafe { bytes_as_os_str(&p.bytes()[0..pos]) };
            Some(prefix)
        } else {
            // The between UEFI prefix and file-path seems to be `/\`
            None
        }
    }
}

pub(crate) fn device_path_to_path(path: &mut device_path::Protocol) -> io::Result<PathBuf> {
    use crate::alloc::{Allocator, Global, Layout};

    let device_path_to_text_handles = common::locate_handles(device_path_to_text::PROTOCOL_GUID)?;
    for handle in device_path_to_text_handles {
        let protocol: NonNull<device_path_to_text::Protocol> =
            match common::open_protocol(handle, device_path_to_text::PROTOCOL_GUID) {
                Ok(x) => x,
                Err(_) => continue,
            };
        let path_ucs2 = unsafe {
            ((*protocol.as_ptr()).convert_device_path_to_text)(
                path,
                r_efi::efi::Boolean::FALSE,
                r_efi::efi::Boolean::FALSE,
            )
        };
        let ucs2_iter = match unsafe { crate::sys_common::wstr::WStrUnits::new(path_ucs2) } {
            None => break,
            Some(x) => x,
        };

        let mut path: String = String::new();
        for w in ucs2_iter {
            // The Returned u16 should be UCS-2 charcters. Thus it should be fine to directly
            // convert to char instead of creating `crate::sys_common::wtf8::CodePoint` first
            let c = char::from_u32(w.get() as u32).ok_or(io::const_io_error!(
                io::ErrorKind::InvalidFilename,
                "Invalid Device Path"
            ))?;
            path.push(c);
        }

        let layout = unsafe {
            Layout::from_size_align_unchecked(crate::mem::size_of::<u16>() * path.len(), 8usize)
        };
        // Deallocate returned UCS-2 String
        unsafe { Global.deallocate(NonNull::new_unchecked(path_ucs2 as *mut u16).cast(), layout) }
        return Ok(PathBuf::from(path));
    }
    Err(
        io::const_io_error!(io::ErrorKind::InvalidData, "failed to convert to text representation",),
    )
}

pub(crate) fn device_path_from_os_str(path: &OsStr) -> io::Result<Box<device_path::Protocol>> {
    let device_path_from_text_handles =
        common::locate_handles(device_path_from_text::PROTOCOL_GUID)?;
    for handle in device_path_from_text_handles {
        let protocol: NonNull<device_path_from_text::Protocol> =
            match common::open_protocol(handle, device_path_from_text::PROTOCOL_GUID) {
                Ok(x) => x,
                Err(_) => continue,
            };
        let device_path = unsafe {
            ((*protocol.as_ptr()).convert_text_to_device_path)(
                common::to_ffi_string(path).as_mut_ptr(),
            )
        };
        return unsafe { Ok(Box::from_raw(device_path)) };
    }
    Err(
        io::const_io_error!(io::ErrorKind::InvalidData, "failed to convert to text representation",),
    )
}
