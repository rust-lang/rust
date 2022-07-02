use crate::env;
use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'\\'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'\\'
}

#[inline]
pub fn parse_prefix(_: &OsStr) -> Option<Prefix<'_>> {
    todo!()
}

pub const MAIN_SEP_STR: &str = "\\";
pub const MAIN_SEP: char = '\\';

pub(crate) fn absolute(path: &Path) -> io::Result<PathBuf> {
    todo!()
}
