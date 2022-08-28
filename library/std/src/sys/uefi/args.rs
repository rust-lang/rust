//! Args related functionality for UEFI. Takes a lot of inspiration of Windows args

use super::common;
use crate::env::current_exe;
use crate::ffi::OsString;
use crate::fmt;
use crate::path::PathBuf;
use crate::sys_common::args::{parse_lp_cmd_line, WStrUnits};
use crate::vec;
use r_efi::efi::protocols::loaded_image;

pub struct Args {
    parsed_args_list: vec::IntoIter<OsString>,
}

// Get the Supplied arguments for loaded image.
// Uses EFI_LOADED_IMAGE_PROTOCOL
pub fn args() -> Args {
    match common::get_current_handle_protocol::<loaded_image::Protocol>(loaded_image::PROTOCOL_GUID)
    {
        Some(x) => {
            let lp_cmd_line = unsafe { (*x.as_ptr()).load_options as *const u16 };
            let parsed_args_list =
                parse_lp_cmd_line(unsafe { WStrUnits::new(lp_cmd_line) }, || {
                    current_exe().map(PathBuf::into_os_string).unwrap_or_else(|_| OsString::new())
                });

            Args { parsed_args_list: parsed_args_list.into_iter() }
        }
        None => Args { parsed_args_list: Vec::new().into_iter() },
    }
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.parsed_args_list.as_slice().fmt(f)
    }
}

impl Iterator for Args {
    type Item = OsString;

    #[inline]
    fn next(&mut self) -> Option<OsString> {
        self.parsed_args_list.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.parsed_args_list.size_hint()
    }
}

impl DoubleEndedIterator for Args {
    #[inline]
    fn next_back(&mut self) -> Option<OsString> {
        self.parsed_args_list.next_back()
    }
}

impl ExactSizeIterator for Args {
    #[inline]
    fn len(&self) -> usize {
        self.parsed_args_list.len()
    }
}
