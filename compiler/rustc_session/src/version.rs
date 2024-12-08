use std::borrow::Cow;
use std::fmt::{self, Display};

use rustc_errors::IntoDiagArg;
use rustc_macros::{Decodable, Encodable, HashStable_Generic, current_rustc_version};

#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(HashStable_Generic)]
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

impl IntoDiagArg for RustcVersion {
    fn into_diag_arg(self) -> rustc_errors::DiagArgValue {
        rustc_errors::DiagArgValue::Str(Cow::Owned(self.to_string()))
    }
}
