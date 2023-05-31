//! The Windows command line is just a string
//! <https://docs.microsoft.com/en-us/archive/blogs/larryosterman/the-windows-command-line-is-just-a-string>
//!
//! This module implements the parsing necessary to turn that string into a list of arguments.

#[cfg(test)]
mod tests;

use crate::ffi::OsString;
use crate::fmt;
use crate::io;
use crate::num::NonZeroU16;
use crate::os::windows::prelude::*;
use crate::path::{Path, PathBuf};
use crate::sys::path::get_long_path;
use crate::sys::process::ensure_no_nuls;
use crate::sys::windows::os::current_exe;
use crate::sys::{c, to_u16s};
use crate::sys_common::wstr::WStrUnits;
use crate::vec;

use crate::iter;

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

pub fn args() -> Args {
    // SAFETY: `GetCommandLineW` returns a pointer to a null terminated UTF-16
    // string so it's safe for `WStrUnits` to use.
    unsafe {
        let lp_cmd_line = c::GetCommandLineW();
        let parsed_args_list = parse_lp_cmd_line(WStrUnits::new(lp_cmd_line), || {
            current_exe().map(PathBuf::into_os_string).unwrap_or_else(|_| OsString::new())
        });

        Args { parsed_args_list: parsed_args_list.into_iter() }
    }
}

/// Implements the Windows command-line argument parsing algorithm.
///
/// Microsoft's documentation for the Windows CLI argument format can be found at
/// <https://docs.microsoft.com/en-us/cpp/cpp/main-function-command-line-args?view=msvc-160#parsing-c-command-line-arguments>
///
/// A more in-depth explanation is here:
/// <https://daviddeley.com/autohotkey/parameters/parameters.htm#WIN>
///
/// Windows includes a function to do command line parsing in shell32.dll.
/// However, this is not used for two reasons:
///
/// 1. Linking with that DLL causes the process to be registered as a GUI application.
/// GUI applications add a bunch of overhead, even if no windows are drawn. See
/// <https://randomascii.wordpress.com/2018/12/03/a-not-called-function-can-cause-a-5x-slowdown/>.
///
/// 2. It does not follow the modern C/C++ argv rules outlined in the first two links above.
///
/// This function was tested for equivalence to the C/C++ parsing rules using an
/// extensive test suite available at
/// <https://github.com/ChrisDenton/winarg/tree/std>.
fn parse_lp_cmd_line<'a, F: Fn() -> OsString>(
    lp_cmd_line: Option<WStrUnits<'a>>,
    exe_name: F,
) -> Vec<OsString> {
    const BACKSLASH: NonZeroU16 = non_zero_u16(b'\\' as u16);
    const QUOTE: NonZeroU16 = non_zero_u16(b'"' as u16);
    const TAB: NonZeroU16 = non_zero_u16(b'\t' as u16);
    const SPACE: NonZeroU16 = non_zero_u16(b' ' as u16);

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
            SPACE | TAB if !in_quotes => break,
            // In all other cases the code unit is taken literally.
            _ => cur.push(w.get()),
        }
    }
    // Skip whitespace.
    code_units.advance_while(|w| w == SPACE || w == TAB);
    ret_val.push(OsString::from_wide(&cur));

    // Parse the arguments according to these rules:
    // * All code units are taken literally except space, tab, quote and backslash.
    // * When not `in_quotes`, space and tab separate arguments. Consecutive spaces and tabs are
    // treated as a single separator.
    // * A space or tab `in_quotes` is taken literally.
    // * A quote toggles `in_quotes` mode unless it's escaped. An escaped quote is taken literally.
    // * A quote can be escaped if preceded by an odd number of backslashes.
    // * If any number of backslashes is immediately followed by a quote then the number of
    // backslashes is halved (rounding down).
    // * Backslashes not followed by a quote are all taken literally.
    // * If `in_quotes` then a quote can also be escaped using another quote
    // (i.e. two consecutive quotes become one literal quote).
    let mut cur = Vec::new();
    let mut in_quotes = false;
    while let Some(w) = code_units.next() {
        match w {
            // If not `in_quotes`, a space or tab ends the argument.
            SPACE | TAB if !in_quotes => {
                ret_val.push(OsString::from_wide(&cur[..]));
                cur.truncate(0);

                // Skip whitespace.
                code_units.advance_while(|w| w == SPACE || w == TAB);
            }
            // Backslashes can escape quotes or backslashes but only if consecutive backslashes are followed by a quote.
            BACKSLASH => {
                let backslash_count = code_units.advance_while(|w| w == BACKSLASH) + 1;
                if code_units.peek() == Some(QUOTE) {
                    cur.extend(iter::repeat(BACKSLASH.get()).take(backslash_count / 2));
                    // The quote is escaped if there are an odd number of backslashes.
                    if backslash_count % 2 == 1 {
                        code_units.next();
                        cur.push(QUOTE.get());
                    }
                } else {
                    // If there is no quote on the end then there is no escaping.
                    cur.extend(iter::repeat(BACKSLASH.get()).take(backslash_count));
                }
            }
            // If `in_quotes` and not backslash escaped (see above) then a quote either
            // unsets `in_quote` or is escaped by another quote.
            QUOTE if in_quotes => match code_units.peek() {
                // Two consecutive quotes when `in_quotes` produces one literal quote.
                Some(QUOTE) => {
                    cur.push(QUOTE.get());
                    code_units.next();
                }
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

pub struct Args {
    parsed_args_list: vec::IntoIter<OsString>,
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.parsed_args_list.as_slice().fmt(f)
    }
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> {
        self.parsed_args_list.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.parsed_args_list.size_hint()
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.parsed_args_list.next_back()
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        self.parsed_args_list.len()
    }
}

#[derive(Debug)]
pub(crate) enum Arg {
    /// Add quotes (if needed)
    Regular(OsString),
    /// Append raw string without quoting
    Raw(OsString),
}

enum Quote {
    // Every arg is quoted
    Always,
    // Whitespace and empty args are quoted
    Auto,
    // Arg appended without any changes (#29494)
    Never,
}

pub(crate) fn append_arg(cmd: &mut Vec<u16>, arg: &Arg, force_quotes: bool) -> io::Result<()> {
    let (arg, quote) = match arg {
        Arg::Regular(arg) => (arg, if force_quotes { Quote::Always } else { Quote::Auto }),
        Arg::Raw(arg) => (arg, Quote::Never),
    };

    // If an argument has 0 characters then we need to quote it to ensure
    // that it actually gets passed through on the command line or otherwise
    // it will be dropped entirely when parsed on the other end.
    ensure_no_nuls(arg)?;
    let arg_bytes = arg.as_os_str_bytes();
    let (quote, escape) = match quote {
        Quote::Always => (true, true),
        Quote::Auto => {
            (arg_bytes.iter().any(|c| *c == b' ' || *c == b'\t') || arg_bytes.is_empty(), true)
        }
        Quote::Never => (false, false),
    };
    if quote {
        cmd.push('"' as u16);
    }

    let mut backslashes: usize = 0;
    for x in arg.encode_wide() {
        if escape {
            if x == '\\' as u16 {
                backslashes += 1;
            } else {
                if x == '"' as u16 {
                    // Add n+1 backslashes to total 2n+1 before internal '"'.
                    cmd.extend((0..=backslashes).map(|_| '\\' as u16));
                }
                backslashes = 0;
            }
        }
        cmd.push(x);
    }

    if quote {
        // Add n backslashes to total 2n before ending '"'.
        cmd.extend((0..backslashes).map(|_| '\\' as u16));
        cmd.push('"' as u16);
    }
    Ok(())
}

pub(crate) fn make_bat_command_line(
    script: &[u16],
    args: &[Arg],
    force_quotes: bool,
) -> io::Result<Vec<u16>> {
    // Set the start of the command line to `cmd.exe /c "`
    // It is necessary to surround the command in an extra pair of quotes,
    // hence the trailing quote here. It will be closed after all arguments
    // have been added.
    let mut cmd: Vec<u16> = "cmd.exe /d /c \"".encode_utf16().collect();

    // Push the script name surrounded by its quote pair.
    cmd.push(b'"' as u16);
    // Windows file names cannot contain a `"` character or end with `\\`.
    // If the script name does then return an error.
    if script.contains(&(b'"' as u16)) || script.last() == Some(&(b'\\' as u16)) {
        return Err(io::const_io_error!(
            io::ErrorKind::InvalidInput,
            "Windows file names may not contain `\"` or end with `\\`"
        ));
    }
    cmd.extend_from_slice(script.strip_suffix(&[0]).unwrap_or(script));
    cmd.push(b'"' as u16);

    // Append the arguments.
    // FIXME: This needs tests to ensure that the arguments are properly
    // reconstructed by the batch script by default.
    for arg in args {
        cmd.push(' ' as u16);
        // Make sure to always quote special command prompt characters, including:
        // * Characters `cmd /?` says require quotes.
        // * `%` for environment variables, as in `%TMP%`.
        // * `|<>` pipe/redirect characters.
        const SPECIAL: &[u8] = b"\t &()[]{}^=;!'+,`~%|<>";
        let force_quotes = match arg {
            Arg::Regular(arg) if !force_quotes => {
                arg.as_os_str_bytes().iter().any(|c| SPECIAL.contains(c))
            }
            _ => force_quotes,
        };
        append_arg(&mut cmd, arg, force_quotes)?;
    }

    // Close the quote we left opened earlier.
    cmd.push(b'"' as u16);

    Ok(cmd)
}

/// Takes a path and tries to return a non-verbatim path.
///
/// This is necessary because cmd.exe does not support verbatim paths.
pub(crate) fn to_user_path(path: &Path) -> io::Result<Vec<u16>> {
    from_wide_to_user_path(to_u16s(path)?)
}
pub(crate) fn from_wide_to_user_path(mut path: Vec<u16>) -> io::Result<Vec<u16>> {
    use crate::ptr;
    use crate::sys::windows::fill_utf16_buf;

    // UTF-16 encoded code points, used in parsing and building UTF-16 paths.
    // All of these are in the ASCII range so they can be cast directly to `u16`.
    const SEP: u16 = b'\\' as _;
    const QUERY: u16 = b'?' as _;
    const COLON: u16 = b':' as _;
    const U: u16 = b'U' as _;
    const N: u16 = b'N' as _;
    const C: u16 = b'C' as _;

    // Early return if the path is too long to remove the verbatim prefix.
    const LEGACY_MAX_PATH: usize = 260;
    if path.len() > LEGACY_MAX_PATH {
        return Ok(path);
    }

    match &path[..] {
        // `\\?\C:\...` => `C:\...`
        [SEP, SEP, QUERY, SEP, _, COLON, SEP, ..] => unsafe {
            let lpfilename = path[4..].as_ptr();
            fill_utf16_buf(
                |buffer, size| c::GetFullPathNameW(lpfilename, size, buffer, ptr::null_mut()),
                |full_path: &[u16]| {
                    if full_path == &path[4..path.len() - 1] {
                        let mut path: Vec<u16> = full_path.into();
                        path.push(0);
                        path
                    } else {
                        path
                    }
                },
            )
        },
        // `\\?\UNC\...` => `\\...`
        [SEP, SEP, QUERY, SEP, U, N, C, SEP, ..] => unsafe {
            // Change the `C` in `UNC\` to `\` so we can get a slice that starts with `\\`.
            path[6] = b'\\' as u16;
            let lpfilename = path[6..].as_ptr();
            fill_utf16_buf(
                |buffer, size| c::GetFullPathNameW(lpfilename, size, buffer, ptr::null_mut()),
                |full_path: &[u16]| {
                    if full_path == &path[6..path.len() - 1] {
                        let mut path: Vec<u16> = full_path.into();
                        path.push(0);
                        path
                    } else {
                        // Restore the 'C' in "UNC".
                        path[6] = b'C' as u16;
                        path
                    }
                },
            )
        },
        // For everything else, leave the path unchanged.
        _ => get_long_path(path, false),
    }
}
