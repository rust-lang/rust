//! Args related functionality for UEFI. Takes a lot of inspiration of Windows args

use super::common;
use crate::env::current_exe;
use crate::ffi::OsString;
use crate::fmt;
use crate::path::PathBuf;
use crate::sync::OnceLock;
use crate::sys_common::wstr::WStrUnits;
use crate::vec;
use r_efi::efi::protocols::loaded_image;

pub struct Args {
    parsed_args_list: vec::IntoIter<OsString>,
}

// Get the Supplied arguments for loaded image.
// Uses EFI_LOADED_IMAGE_PROTOCOL
pub fn args() -> Args {
    static ARGUMENTS: OnceLock<Vec<OsString>> = OnceLock::new();
    // Caching the arguments the first time they are parsed.
    let vec_args = ARGUMENTS.get_or_init(|| {
        match common::get_current_handle_protocol::<loaded_image::Protocol>(
            loaded_image::PROTOCOL_GUID,
        ) {
            Some(x) => {
                let lp_cmd_line = unsafe { (*x.as_ptr()).load_options as *const u16 };
                parse_lp_cmd_line(unsafe { WStrUnits::new(lp_cmd_line) }, || {
                    current_exe().map(PathBuf::into_os_string).unwrap_or_else(|_| OsString::new())
                })
            }
            None => Vec::new(),
        }
    });
    Args { parsed_args_list: vec_args.clone().into_iter() }
}

pub(crate) fn parse_lp_cmd_line<'a, F: Fn() -> OsString>(
    _lp_cmd_line: Option<WStrUnits<'a>>,
    _exe_name: F,
) -> Vec<OsString> {
    todo!()
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
