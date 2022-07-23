use super::ffi::OsStrExt;
use super::raw::protocols::{device_path, device_path_from_text, device_path_to_text};
use crate::alloc::{Allocator, Global, Layout};
use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf};
use crate::ptr::NonNull;
use crate::sys_common::ucs2;

#[unstable(feature = "uefi_std", issue = "none")]
impl TryFrom<NonNull<device_path::Protocol>> for PathBuf {
    type Error = crate::io::Error;

    fn try_from(
        value: NonNull<super::raw::protocols::device_path::Protocol>,
    ) -> Result<Self, Self::Error> {
        let device_path_to_text_handles =
            super::env::locate_handles(device_path_to_text::PROTOCOL_GUID)?;
        for handle in device_path_to_text_handles {
            let protocol: NonNull<device_path_to_text::Protocol> =
                match super::env::open_protocol(handle, device_path_to_text::PROTOCOL_GUID) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
            let path_ucs2 = unsafe {
                ((*protocol.as_ptr()).convert_device_path_to_text)(
                    value.as_ptr(),
                    super::raw::Boolean::FALSE,
                    super::raw::Boolean::FALSE,
                )
            };
            let ucs2_iter = match unsafe { ucs2::Ucs2Units::new(path_ucs2) } {
                None => break,
                Some(x) => x,
            };
            let mut len = 0;
            let mut path = String::new();
            for c in ucs2_iter {
                let ch = char::from(ucs2::Ucs2Char::from_u16(u16::from(c)));
                path.push(ch);
                len += 1;
            }

            let layout =
                Layout::from_size_align(crate::mem::size_of::<u16>() * len, 8usize).unwrap();
            // Deallocate returned UCS-2 String
            unsafe {
                Global.deallocate(NonNull::new_unchecked(path_ucs2 as *mut u16).cast(), layout)
            }
            return Ok(PathBuf::from(path));
        }
        Err(crate::io::Error::new(
            crate::io::ErrorKind::InvalidData,
            "Failed to Convert to text representation",
        ))
    }
}

impl TryFrom<&Path> for DevicePath {
    type Error = crate::io::Error;

    fn try_from(value: &Path) -> Result<Self, Self::Error> {
        DevicePath::try_from(value.as_os_str())
    }
}

impl TryFrom<&OsStr> for DevicePath {
    type Error = crate::io::Error;

    fn try_from(value: &OsStr) -> Result<Self, Self::Error> {
        let device_path_from_text_handles =
            super::env::locate_handles(device_path_from_text::PROTOCOL_GUID)?;
        for handle in device_path_from_text_handles {
            let protocol: NonNull<device_path_from_text::Protocol> =
                match super::env::open_protocol(handle, device_path_from_text::PROTOCOL_GUID) {
                    Ok(x) => x,
                    Err(_) => continue,
                };
            let device_path = unsafe {
                ((*protocol.as_ptr()).convert_text_to_device_path)(
                    value.to_ffi_string().as_mut_ptr(),
                )
            };
            let device_path = match NonNull::new(device_path) {
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::Uncategorized,
                        "Null DevicePath Returned",
                    ));
                }
                Some(x) => x,
            };

            let layout =
                Layout::from_size_align(crate::mem::size_of::<device_path::Protocol>(), 8usize)
                    .unwrap();
            return Ok(DevicePath::new(device_path, layout));
        }
        Err(crate::io::Error::new(
            crate::io::ErrorKind::InvalidData,
            "Failed to Convert to text representation",
        ))
    }
}

pub(crate) type DevicePath = super::raw::VariableSizeType<device_path::Protocol>;
