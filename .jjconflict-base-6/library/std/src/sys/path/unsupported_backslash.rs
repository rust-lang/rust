#![forbid(unsafe_op_in_unsafe_fn)]
use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};
use crate::sys::unsupported;

path_separator_bytes!(b'\\');

#[inline]
pub const fn is_verbatim_sep(b: u8) -> bool {
    is_sep_byte(b)
}

pub fn parse_prefix(_: &OsStr) -> Option<Prefix<'_>> {
    None
}

pub const HAS_PREFIXES: bool = true;

pub(crate) fn absolute(_path: &Path) -> io::Result<PathBuf> {
    unsupported()
}

pub(crate) fn is_absolute(path: &Path) -> bool {
    path.has_root() && path.prefix().is_some()
}
