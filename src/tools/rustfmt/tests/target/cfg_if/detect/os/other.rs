//! Other operating systems

use crate::detect::Feature;

/// Performs run-time feature detection.
#[inline]
pub fn check_for(_x: Feature) -> bool {
    false
}
