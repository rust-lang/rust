//! Small helper functions used inside `sys`.
//!
//! If any of these have uses outside of `sys`, please move them to a different
//! module.

#[cfg_attr(not(target_os = "linux"), allow(unused))] // Not used on all platforms.
mod small_c_string;
#[cfg_attr(not(target_os = "windows"), allow(unused))] // Not used on all platforms.
mod wstr;

#[cfg(test)]
mod tests;

#[cfg_attr(not(target_os = "linux"), allow(unused))] // Not used on all platforms.
pub use small_c_string::{run_path_with_cstr, run_with_cstr};
#[cfg_attr(not(target_os = "windows"), allow(unused))] // Not used on all platforms.
pub use wstr::WStrUnits;

/// Computes `(value*numerator)/denom` without overflow, as long as both
/// `numerator*denom` and the overall result fit into `u64` (which is the case
/// for our time conversions).
#[cfg_attr(not(target_os = "windows"), allow(unused))] // Not used on all platforms.
pub fn mul_div_u64(value: u64, numerator: u64, denom: u64) -> u64 {
    let q = value / denom;
    let r = value % denom;
    // Decompose value as (value/denom*denom + value%denom),
    // substitute into (value*numerator)/denom and simplify.
    // r < denom, so (denom*numerator) is the upper bound of (r*numerator)
    q * numerator + r * numerator / denom
}

#[cfg_attr(not(target_os = "linux"), allow(unused))] // Not used on all platforms.
pub fn ignore_notfound<T>(result: crate::io::Result<T>) -> crate::io::Result<()> {
    match result {
        Err(err) if err.kind() == crate::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Ok(()),
        Err(err) => Err(err),
    }
}
