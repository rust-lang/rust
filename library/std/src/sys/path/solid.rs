#![forbid(unsafe_op_in_unsafe_fn)]
use crate::ffi::{CStr, CString, OsStr};
use crate::io;
use crate::os::solid::ffi::OsStrExt;
use crate::path::{Path, PathBuf, Prefix};
use crate::sys::unsupported;

path_separator_bytes!(b'\\');

pub fn cstr(path: &Path) -> io::Result<CString> {
    let path = path.as_os_str().as_bytes();

    if !path.starts_with(br"\") {
        // Relative paths aren't supported
        return Err(crate::io::const_error!(
            crate::io::ErrorKind::Unsupported,
            "relative path is not supported on this platform",
        ));
    }

    // Apply the thread-safety wrapper
    const SAFE_PREFIX: &[u8] = br"\TS";
    let wrapped_path = [SAFE_PREFIX, &path, &[0]].concat();

    CString::from_vec_with_nul(wrapped_path).map_err(|_| {
        crate::io::const_error!(io::ErrorKind::InvalidInput, "path provided contains a nul byte")
    })
}

#[inline]
pub const fn is_verbatim_sep(b: u8) -> bool {
    is_sep_byte(b)
}

pub fn parse_prefix(_: &OsStr) -> Option<Prefix<'_>> {
    None
}

pub const HAS_PREFIXES: bool = true;

#[inline]
pub fn with_native_path<T>(path: &Path, f: &dyn Fn(&CStr) -> io::Result<T>) -> io::Result<T> {
    let path = cstr(path)?;
    f(&path)
}

pub(crate) fn absolute(_path: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub(crate) fn is_absolute(path: &Path) -> bool {
    path.has_root() && path.prefix().is_some()
}
