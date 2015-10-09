pub use imp::path as imp;

pub mod traits {
    pub use super::PathInfo as sys_PathInfo;
}

pub mod prelude {
    pub use super::imp::PathInfo;
    pub use super::traits::*;
    pub use super::Prefix;
}

use os_str::prelude::*;

pub trait PathInfo {
    fn is_sep_byte(b: u8) -> bool;
    fn is_verbatim_sep(b: u8) -> bool;

    const PREFIX_IMP: bool;
    fn parse_prefix(s: &OsStr) -> Option<Prefix>;

    const MAIN_SEP_STR: &'static str;
    const MAIN_SEP: char;
}

pub enum Prefix<'a> {
    Verbatim(&'a OsStr),
    VerbatimUNC(&'a OsStr, &'a OsStr),
    VerbatimDisk(u8),
    DeviceNS(&'a OsStr),
    UNC(&'a OsStr, &'a OsStr),
    Disk(u8),
}
