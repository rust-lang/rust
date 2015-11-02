use path;
use ffi::OsStr;

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'/'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'/'
}

pub const PREFIX_IMP: bool = false;

pub fn parse_prefix(_: &OsStr) -> Option<path::Prefix> {
    None
}

pub const MAIN_SEP_STR: &'static str = "/";
pub const MAIN_SEP: char = '/';
