//! Implementation of path from UEFI. Mostly just copying Windows Implementation

use crate::env;
use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};

pub const MAIN_SEP_STR: &str = "\\";
pub const MAIN_SEP: char = '\\';

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'\\'
}

#[inline]
pub fn parse_prefix(_: &OsStr) -> Option<Prefix<'_>> {
    todo!()
}

pub(crate) fn absolute(path: &Path) -> io::Result<PathBuf> {
    todo!()
}
