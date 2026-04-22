use super::{ErrorKind, OsFunctions, RawOsError};
use crate::fmt;

/// # Safety
///
/// The provided reference must point to data that is entirely constant; it must
/// not be created during runtime.
#[inline]
pub(super) unsafe fn set_functions(f: &'static OsFunctions) {
    // FIXME: externally implementable items may allow for weak linkage, allowing
    // these methods to be overridden even when atomic pointers are not supported.
}

#[inline]
pub(super) fn format_os_error(errno: RawOsError, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    let f = OsFunctions::DEFAULT;
    (f.format_os_error)(errno, fmt)
}

#[inline]
pub(super) fn decode_error_kind(errno: RawOsError) -> ErrorKind {
    let f = OsFunctions::DEFAULT;
    (f.decode_error_kind)(errno)
}

#[inline]
pub(super) fn is_interrupted(errno: RawOsError) -> bool {
    let f = OsFunctions::DEFAULT;
    (f.is_interrupted)(errno)
}
