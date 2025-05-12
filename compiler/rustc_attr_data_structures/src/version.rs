use std::fmt::{self, Display};

use rustc_macros::{
    Decodable, Encodable, HashStable_Generic, PrintAttribute, current_rustc_version,
};

use crate::PrintAttribute;

#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(HashStable_Generic, PrintAttribute)]
pub struct RustcVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl RustcVersion {
    pub const CURRENT: Self = current_rustc_version!();
}

impl Display for RustcVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}
