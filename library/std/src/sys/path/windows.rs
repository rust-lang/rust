use crate::ffi::{OsStr, OsString};
use crate::path::{Path, PathBuf};
use crate::sys::api::utf16;
use crate::sys::pal::{c, fill_utf16_buf, os2path, to_u16s};
use crate::{io, ptr};

#[cfg(test)]
mod tests;

pub use super::windows_prefix::parse_prefix;

pub const MAIN_SEP_STR: &str = "\\";
pub const MAIN_SEP: char = '\\';

/// A null terminated wide string.
#[repr(transparent)]
pub struct WCStr([u16]);

impl WCStr {
    /// Convert a slice to a WCStr without checks.
    ///
    /// Though it is memory safe, the slice should also not contain interior nulls
    /// as this may lead to unwanted truncation.
    ///
    /// # Safety
    ///
    /// The slice must end in a null.
    pub unsafe fn from_wchars_with_null_unchecked(s: &[u16]) -> &Self {
        unsafe { &*(s as *const [u16] as *const Self) }
    }

    pub fn as_ptr(&self) -> *const u16 {
        self.0.as_ptr()
    }

    pub fn count_bytes(&self) -> usize {
        self.0.len()
    }
}

#[inline]
pub fn with_native_path<T>(path: &Path, f: &dyn Fn(&WCStr) -> io::Result<T>) -> io::Result<T> {
    let path = maybe_verbatim(path)?;
    // SAFETY: maybe_verbatim returns null-terminated strings
    let path = unsafe { WCStr::from_wchars_with_null_unchecked(&path) };
    f(path)
}

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'\\'
}

pub fn is_verbatim(path: &[u16]) -> bool {
    path.starts_with(utf16!(r"\\?\")) || path.starts_with(utf16!(r"\??\"))
}

/// Returns true if `path` looks like a lone filename.
pub(crate) fn is_file_name(path: &OsStr) -> bool {
    !path.as_encoded_bytes().iter().copied().any(is_sep_byte)
}
pub(crate) fn has_trailing_slash(path: &OsStr) -> bool {
    let is_verbatim = path.as_encoded_bytes().starts_with(br"\\?\");
    let is_separator = if is_verbatim { is_verbatim_sep } else { is_sep_byte };
    if let Some(&c) = path.as_encoded_bytes().last() { is_separator(c) } else { false }
}

/// Appends a suffix to a path.
///
/// Can be used to append an extension without removing an existing extension.
pub(crate) fn append_suffix(path: PathBuf, suffix: &OsStr) -> PathBuf {
    let mut path = OsString::from(path);
    path.push(suffix);
    path.into()
}

/// Returns a UTF-16 encoded path capable of bypassing the legacy `MAX_PATH` limits.
///
/// This path may or may not have a verbatim prefix.
pub(crate) fn maybe_verbatim(path: &Path) -> io::Result<Vec<u16>> {
    let path = to_u16s(path)?;
    get_long_path(path, true)
}

/// Gets a normalized absolute path that can bypass path length limits.
///
/// Setting prefer_verbatim to true suggests a stronger preference for verbatim
/// paths even when not strictly necessary. This allows the Windows API to avoid
/// repeating our work. However, if the path may be given back to users or
/// passed to other application then it's preferable to use non-verbatim paths
/// when possible. Non-verbatim paths are better understood by users and handled
/// by more software.
pub(crate) fn get_long_path(mut path: Vec<u16>, prefer_verbatim: bool) -> io::Result<Vec<u16>> {
    // Normally the MAX_PATH is 260 UTF-16 code units (including the NULL).
    // However, for APIs such as CreateDirectory[1], the limit is 248.
    //
    // [1]: https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createdirectorya#parameters
    const LEGACY_MAX_PATH: usize = 248;
    // UTF-16 encoded code points, used in parsing and building UTF-16 paths.
    // All of these are in the ASCII range so they can be cast directly to `u16`.
    const SEP: u16 = b'\\' as _;
    const ALT_SEP: u16 = b'/' as _;
    const QUERY: u16 = b'?' as _;
    const COLON: u16 = b':' as _;
    const DOT: u16 = b'.' as _;
    const U: u16 = b'U' as _;
    const N: u16 = b'N' as _;
    const C: u16 = b'C' as _;

    // \\?\
    const VERBATIM_PREFIX: &[u16] = &[SEP, SEP, QUERY, SEP];
    // \??\
    const NT_PREFIX: &[u16] = &[SEP, QUERY, QUERY, SEP];
    // \\?\UNC\
    const UNC_PREFIX: &[u16] = &[SEP, SEP, QUERY, SEP, U, N, C, SEP];

    if path.starts_with(VERBATIM_PREFIX) || path.starts_with(NT_PREFIX) || path == [0] {
        // Early return for paths that are already verbatim or empty.
        return Ok(path);
    } else if path.len() < LEGACY_MAX_PATH {
        // Early return if an absolute path is less < 260 UTF-16 code units.
        // This is an optimization to avoid calling `GetFullPathNameW` unnecessarily.
        match path.as_slice() {
            // Starts with `D:`, `D:\`, `D:/`, etc.
            // Does not match if the path starts with a `\` or `/`.
            [drive, COLON, 0] | [drive, COLON, SEP | ALT_SEP, ..]
                if *drive != SEP && *drive != ALT_SEP =>
            {
                return Ok(path);
            }
            // Starts with `\\`, `//`, etc
            [SEP | ALT_SEP, SEP | ALT_SEP, ..] => return Ok(path),
            _ => {}
        }
    }

    // Firstly, get the absolute path using `GetFullPathNameW`.
    // https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfullpathnamew
    let lpfilename = path.as_ptr();
    fill_utf16_buf(
        // SAFETY: `fill_utf16_buf` ensures the `buffer` and `size` are valid.
        // `lpfilename` is a pointer to a null terminated string that is not
        // invalidated until after `GetFullPathNameW` returns successfully.
        |buffer, size| unsafe { c::GetFullPathNameW(lpfilename, size, buffer, ptr::null_mut()) },
        |mut absolute| {
            path.clear();

            // Only prepend the prefix if needed.
            if prefer_verbatim || absolute.len() + 1 >= LEGACY_MAX_PATH {
                // Secondly, add the verbatim prefix. This is easier here because we know the
                // path is now absolute and fully normalized (e.g. `/` has been changed to `\`).
                let prefix = match absolute {
                    // C:\ => \\?\C:\
                    [_, COLON, SEP, ..] => VERBATIM_PREFIX,
                    // \\.\ => \\?\
                    [SEP, SEP, DOT, SEP, ..] => {
                        absolute = &absolute[4..];
                        VERBATIM_PREFIX
                    }
                    // Leave \\?\ and \??\ as-is.
                    [SEP, SEP, QUERY, SEP, ..] | [SEP, QUERY, QUERY, SEP, ..] => &[],
                    // \\ => \\?\UNC\
                    [SEP, SEP, ..] => {
                        absolute = &absolute[2..];
                        UNC_PREFIX
                    }
                    // Anything else we leave alone.
                    _ => &[],
                };

                path.reserve_exact(prefix.len() + absolute.len() + 1);
                path.extend_from_slice(prefix);
            } else {
                path.reserve_exact(absolute.len() + 1);
            }
            path.extend_from_slice(absolute);
            path.push(0);
        },
    )?;
    Ok(path)
}

/// Make a Windows path absolute.
pub(crate) fn absolute(path: &Path) -> io::Result<PathBuf> {
    let path = path.as_os_str();
    let prefix = parse_prefix(path);
    // Verbatim paths should not be modified.
    if prefix.map(|x| x.is_verbatim()).unwrap_or(false) {
        // NULs in verbatim paths are rejected for consistency.
        if path.as_encoded_bytes().contains(&0) {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "strings passed to WinAPI cannot contain NULs",
            ));
        }
        return Ok(path.to_owned().into());
    }

    let path = to_u16s(path)?;
    let lpfilename = path.as_ptr();
    fill_utf16_buf(
        // SAFETY: `fill_utf16_buf` ensures the `buffer` and `size` are valid.
        // `lpfilename` is a pointer to a null terminated string that is not
        // invalidated until after `GetFullPathNameW` returns successfully.
        |buffer, size| unsafe { c::GetFullPathNameW(lpfilename, size, buffer, ptr::null_mut()) },
        os2path,
    )
}

pub(crate) fn is_absolute(path: &Path) -> bool {
    path.has_root() && path.prefix().is_some()
}

/// Test that the path is absolute, fully qualified and unchanged when processed by the Windows API.
///
/// For example:
///
/// - `C:\path\to\file` will return true.
/// - `C:\path\to\nul` returns false because the Windows API will convert it to \\.\NUL
/// - `C:\path\to\..\file` returns false because it will be resolved to `C:\path\file`.
///
/// This is a useful property because it means the path can be converted from and to and verbatim
/// path just by changing the prefix.
pub(crate) fn is_absolute_exact(path: &[u16]) -> bool {
    // This is implemented by checking that passing the path through
    // GetFullPathNameW does not change the path in any way.

    // Windows paths are limited to i16::MAX length
    // though the API here accepts a u32 for the length.
    if path.is_empty() || path.len() > u32::MAX as usize || path.last() != Some(&0) {
        return false;
    }
    // The path returned by `GetFullPathNameW` must be the same length as the
    // given path, otherwise they're not equal.
    let buffer_len = path.len();
    let mut new_path = Vec::with_capacity(buffer_len);
    let result = unsafe {
        c::GetFullPathNameW(
            path.as_ptr(),
            new_path.capacity() as u32,
            new_path.as_mut_ptr(),
            crate::ptr::null_mut(),
        )
    };
    // Note: if non-zero, the returned result is the length of the buffer without the null termination
    if result == 0 || result as usize != buffer_len - 1 {
        false
    } else {
        // SAFETY: `GetFullPathNameW` initialized `result` bytes and does not exceed `nBufferLength - 1` (capacity).
        unsafe {
            new_path.set_len((result as usize) + 1);
        }
        path == &new_path
    }
}
