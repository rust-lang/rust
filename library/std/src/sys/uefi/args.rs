//! Args related functionality for UEFI. Takes a lot of inspiration of Windows args

use super::common;
use crate::env::current_exe;
use crate::ffi::OsString;
use crate::fmt;
use crate::num::NonZeroU16;
use crate::os::uefi::ffi::OsStringExt;
use crate::path::PathBuf;
use crate::sync::OnceLock;
use crate::sys_common::wstr::WStrUnits;
use crate::vec;
use r_efi::efi::protocols::loaded_image;

/// This is the const equivalent to `NonZeroU16::new(n).unwrap()`
///
/// FIXME: This can be removed once `Option::unwrap` is stably const.
/// See the `const_option` feature (#67441).
const fn non_zero_u16(n: u16) -> NonZeroU16 {
    match NonZeroU16::new(n) {
        Some(n) => n,
        None => panic!("called `unwrap` on a `None` value"),
    }
}

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

/// Implements the UEFI command-line argument parsing algorithm.
///
/// While this sounds good in theory, I have not really found any concrete implementation of
/// argument parsing in UEFI. Thus I have created thisimplementation based on what is defined in
/// Section 3.2 of [UEFI Shell Specification](https://uefi.org/sites/default/files/resources/UEFI_Shell_Spec_2_0.pdf)
pub(crate) fn parse_lp_cmd_line<'a, F: Fn() -> OsString>(
    lp_cmd_line: Option<WStrUnits<'a>>,
    exe_name: F,
) -> Vec<OsString> {
    const QUOTE: NonZeroU16 = non_zero_u16(b'"' as u16);
    const SPACE: NonZeroU16 = non_zero_u16(b' ' as u16);
    const CARET: NonZeroU16 = non_zero_u16(b'^' as u16);

    let mut ret_val = Vec::new();
    // If the cmd line pointer is null or it points to an empty string then
    // return the name of the executable as argv[0].
    if lp_cmd_line.as_ref().and_then(|cmd| cmd.peek()).is_none() {
        ret_val.push(exe_name());
        return ret_val;
    }
    let mut code_units = lp_cmd_line.unwrap();

    // The executable name at the beginning is special.
    let mut in_quotes = false;
    let mut cur = Vec::new();
    for w in &mut code_units {
        match w {
            // A quote mark always toggles `in_quotes` no matter what because
            // there are no escape characters when parsing the executable name.
            QUOTE => in_quotes = !in_quotes,
            // If not `in_quotes` then whitespace ends argv[0].
            SPACE if !in_quotes => break,
            // In all other cases the code unit is taken literally.
            _ => cur.push(w.get()),
        }
    }

    // Skip whitespace.
    code_units.advance_while(|w| w == SPACE);
    ret_val.push(OsString::from_wide(&cur));

    // Parse the arguments according to these rules:
    // * All code units are taken literally except space, quote and caret.
    // * When not `in_quotes`, space separate arguments. Consecutive spaces are
    // treated as a single separator.
    // * A space `in_quotes` is taken literally.
    // * A quote toggles `in_quotes` mode unless it's escaped. An escaped quote is taken literally.
    // * A quote can be escaped if preceded by caret.
    // * A caret can be escaped if preceded by caret.
    let mut cur = Vec::new();
    let mut in_quotes = false;
    while let Some(w) = code_units.next() {
        match w {
            // If not `in_quotes`, a space or tab ends the argument.
            SPACE if !in_quotes => {
                ret_val.push(OsString::from_wide(&cur[..]));
                cur.truncate(0);

                // Skip whitespace.
                code_units.advance_while(|w| w == SPACE);
            }
            // Caret can escape quotes or carets
            CARET if in_quotes => {
                if let Some(x) = code_units.next() {
                    cur.push(x.get())
                }
            }
            // If `in_quotes` and not backslash escaped (see above) then a quote either
            // unsets `in_quote` or is escaped by another quote.
            QUOTE if in_quotes => match code_units.peek() {
                // Otherwise set `in_quotes`.
                Some(_) => in_quotes = false,
                // The end of the command line.
                // Push `cur` even if empty, which we do by breaking while `in_quotes` is still set.
                None => break,
            },
            // If not `in_quotes` and not BACKSLASH escaped (see above) then a quote sets `in_quote`.
            QUOTE => in_quotes = true,
            // Everything else is always taken literally.
            _ => cur.push(w.get()),
        }
    }
    // Push the final argument, if any.
    if !cur.is_empty() || in_quotes {
        ret_val.push(OsString::from_wide(&cur[..]));
    }
    ret_val
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
