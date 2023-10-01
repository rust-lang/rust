use super::unsupported;
use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};

pub const MAIN_SEP_STR: &str = "\\";
pub const MAIN_SEP: char = '\\';

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'\\'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'\\'
}

pub fn parse_prefix(_p: &OsStr) -> Option<Prefix<'_>> {
    None
}

pub(crate) fn absolute(_path: &Path) -> io::Result<PathBuf> {
    unsupported()
}
