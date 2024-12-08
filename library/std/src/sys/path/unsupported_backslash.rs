#![forbid(unsafe_op_in_unsafe_fn)]
use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};
use crate::sys::unsupported;

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'\\'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'\\'
}

pub fn parse_prefix(_: &OsStr) -> Option<Prefix<'_>> {
    None
}

pub const MAIN_SEP_STR: &str = "\\";
pub const MAIN_SEP: char = '\\';

pub(crate) fn absolute(_path: &Path) -> io::Result<PathBuf> {
    unsupported()
}
