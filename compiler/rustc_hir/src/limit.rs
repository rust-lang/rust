use std::fmt;
use std::ops::{Div, Mul};

use rustc_error_messages::{DiagArgValue, IntoDiagArg};
use rustc_macros::{Decodable, Encodable, HashStable_Generic};

/// New-type wrapper around `usize` for representing limits. Ensures that comparisons against
/// limits are consistent throughout the compiler.
#[derive(Clone, Copy, Debug, HashStable_Generic, Encodable, Decodable)]
pub struct Limit(pub usize);

impl Limit {
    /// Create a new limit from a `usize`.
    pub fn new(value: usize) -> Self {
        Limit(value)
    }

    /// Create a new unlimited limit.
    pub fn unlimited() -> Self {
        Limit(usize::MAX)
    }

    /// Check that `value` is within the limit. Ensures that the same comparisons are used
    /// throughout the compiler, as mismatches can cause ICEs, see #72540.
    #[inline]
    pub fn value_within_limit(&self, value: usize) -> bool {
        value <= self.0
    }
}

impl From<usize> for Limit {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Div<usize> for Limit {
    type Output = Limit;

    fn div(self, rhs: usize) -> Self::Output {
        Limit::new(self.0 / rhs)
    }
}

impl Mul<usize> for Limit {
    type Output = Limit;

    fn mul(self, rhs: usize) -> Self::Output {
        Limit::new(self.0 * rhs)
    }
}

impl IntoDiagArg for Limit {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_string().into_diag_arg(&mut None)
    }
}
