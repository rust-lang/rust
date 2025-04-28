use r_efi::protocols::loaded_image;

pub use super::common::Args;
use crate::env::current_exe;
use crate::ffi::OsString;
use crate::iter::Iterator;
use crate::sys::pal::helpers;

pub fn args() -> Args {
    let lazy_current_exe = || Vec::from([current_exe().map(Into::into).unwrap_or_default()]);

    // Each loaded image has an image handle that supports `EFI_LOADED_IMAGE_PROTOCOL`. Thus, this
    // will never fail.
    let protocol =
        helpers::image_handle_protocol::<loaded_image::Protocol>(loaded_image::PROTOCOL_GUID)
            .unwrap();

    let lp_size = unsafe { (*protocol.as_ptr()).load_options_size } as usize;
    // Break if we are sure that it cannot be UTF-16
    if lp_size < size_of::<u16>() || lp_size % size_of::<u16>() != 0 {
        return Args::new(lazy_current_exe());
    }
    let lp_size = lp_size / size_of::<u16>();

    let lp_cmd_line = unsafe { (*protocol.as_ptr()).load_options as *const u16 };
    if !lp_cmd_line.is_aligned() {
        return Args::new(lazy_current_exe());
    }
    let lp_cmd_line = unsafe { crate::slice::from_raw_parts(lp_cmd_line, lp_size) };

    Args::new(parse_lp_cmd_line(lp_cmd_line).unwrap_or_else(lazy_current_exe))
}

/// Implements the UEFI command-line argument parsing algorithm.
///
/// This implementation is based on what is defined in Section 3.4 of
/// [UEFI Shell Specification](https://uefi.org/sites/default/files/resources/UEFI_Shell_Spec_2_0.pdf)
///
/// Returns None in the following cases:
/// - Invalid UTF-16 (unpaired surrogate)
/// - Empty/improper arguments
fn parse_lp_cmd_line(code_units: &[u16]) -> Option<Vec<OsString>> {
    const QUOTE: char = '"';
    const SPACE: char = ' ';
    const CARET: char = '^';
    const NULL: char = '\0';

    let mut ret_val = Vec::new();
    let mut code_units_iter = char::decode_utf16(code_units.iter().cloned()).peekable();

    // The executable name at the beginning is special.
    let mut in_quotes = false;
    let mut cur = String::new();
    while let Some(w) = code_units_iter.next() {
        let w = w.ok()?;
        match w {
            // break on NULL
            NULL => break,
            // A quote mark always toggles `in_quotes` no matter what because
            // there are no escape characters when parsing the executable name.
            QUOTE => in_quotes = !in_quotes,
            // If not `in_quotes` then whitespace ends argv[0].
            SPACE if !in_quotes => break,
            // In all other cases the code unit is taken literally.
            _ => cur.push(w),
        }
    }

    // If exe name is missing, the cli args are invalid
    if cur.is_empty() {
        return None;
    }

    ret_val.push(OsString::from(cur));
    // Skip whitespace.
    while code_units_iter.next_if_eq(&Ok(SPACE)).is_some() {}

    // Parse the arguments according to these rules:
    // * All code units are taken literally except space, quote and caret.
    // * When not `in_quotes`, space separate arguments. Consecutive spaces are
    // treated as a single separator.
    // * A space `in_quotes` is taken literally.
    // * A quote toggles `in_quotes` mode unless it's escaped. An escaped quote is taken literally.
    // * A quote can be escaped if preceded by caret.
    // * A caret can be escaped if preceded by caret.
    let mut cur = String::new();
    let mut in_quotes = false;
    while let Some(w) = code_units_iter.next() {
        let w = w.ok()?;
        match w {
            // break on NULL
            NULL => break,
            // If not `in_quotes`, a space or tab ends the argument.
            SPACE if !in_quotes => {
                ret_val.push(OsString::from(&cur[..]));
                cur.truncate(0);

                // Skip whitespace.
                while code_units_iter.next_if_eq(&Ok(SPACE)).is_some() {}
            }
            // Caret can escape quotes or carets
            CARET if in_quotes => {
                if let Some(x) = code_units_iter.next() {
                    cur.push(x.ok()?);
                }
            }
            // If quote then flip `in_quotes`
            QUOTE => in_quotes = !in_quotes,
            // Everything else is always taken literally.
            _ => cur.push(w),
        }
    }
    // Push the final argument, if any.
    if !cur.is_empty() || in_quotes {
        ret_val.push(OsString::from(cur));
    }
    Some(ret_val)
}
