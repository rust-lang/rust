use path as sys;
use os_str::prelude::*;

pub struct PathInfo(());

impl sys::PathInfo for PathInfo {
    #[inline]
    fn is_sep_byte(b: u8) -> bool {
        b == b'/'
    }

    #[inline]
    fn is_verbatim_sep(b: u8) -> bool {
        b == b'/'
    }

    const PREFIX_IMP: bool = false;

    fn parse_prefix(s: &OsStr) -> Option<sys::Prefix> {
        None
    }

    const MAIN_SEP_STR: &'static str = "/";
    const MAIN_SEP: char = '/';
}
