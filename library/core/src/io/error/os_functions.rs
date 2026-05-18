use super::{ErrorKind, OsFunctions, RawOsError};
use crate::fmt;

/// # Safety
///
/// The provided reference must point to data that is entirely constant; it must
/// not be created during runtime.
#[inline]
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub unsafe fn set_functions(f: &'static OsFunctions) {
    // FIXME: externally implementable items may allow for weak linkage, allowing
    // these methods to be overridden even when atomic pointers are not supported.
}

#[inline]
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn format_os_error(errno: RawOsError, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    let f = OsFunctions::DEFAULT;
    (f.format_os_error)(errno, fmt)
}

#[inline]
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn decode_error_kind(errno: RawOsError) -> ErrorKind {
    let f = OsFunctions::DEFAULT;
    (f.decode_error_kind)(errno)
}

#[inline]
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn is_interrupted(errno: RawOsError) -> bool {
    let f = OsFunctions::DEFAULT;
    (f.is_interrupted)(errno)
}
