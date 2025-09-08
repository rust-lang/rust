//! Platform-independent platform abstraction
//!
//! This is the platform-independent portion of the standard library's
//! platform abstraction layer, whereas `std::sys` is the
//! platform-specific portion.
//!
//! The relationship between `std::sys_common`, `std::sys` and the
//! rest of `std` is complex, with dependencies going in all
//! directions: `std` depending on `sys_common`, `sys_common`
//! depending on `sys`, and `sys` depending on `sys_common` and `std`.
//! This is because `sys_common` not only contains platform-independent code,
//! but also code that is shared between the different platforms in `sys`.
//! Ideally all that shared code should be moved to `sys::common`,
//! and the dependencies between `std`, `sys_common` and `sys` all would form a DAG.
//! Progress on this is tracked in #84187.

#![allow(missing_docs)]
#![allow(missing_debug_implementations)]

#[cfg(test)]
mod tests;

pub mod wstr;

// common error constructors

/// A trait for viewing representations from std types
#[doc(hidden)]
#[allow(dead_code)] // not used on all platforms
pub trait AsInner<Inner: ?Sized> {
    fn as_inner(&self) -> &Inner;
}

/// A trait for viewing representations from std types
#[doc(hidden)]
#[allow(dead_code)] // not used on all platforms
pub trait AsInnerMut<Inner: ?Sized> {
    fn as_inner_mut(&mut self) -> &mut Inner;
}

/// A trait for extracting representations from std types
#[doc(hidden)]
pub trait IntoInner<Inner> {
    fn into_inner(self) -> Inner;
}

/// A trait for creating std types from internal representations
#[doc(hidden)]
pub trait FromInner<Inner> {
    fn from_inner(inner: Inner) -> Self;
}

// Computes (value*numerator)/denom without overflow, as long as both (numerator*denom) and the
// overall result fit into i64 (which is the case for our time conversions).
#[allow(dead_code)] // not used on all platforms
pub fn mul_div_u64(value: u64, numerator: u64, denom: u64) -> u64 {
    let q = value / denom;
    let r = value % denom;
    // Decompose value as (value/denom*denom + value%denom),
    // substitute into (value*numerator)/denom and simplify.
    // r < denom, so (denom*numerator) is the upper bound of (r*numerator)
    q * numerator + r * numerator / denom
}

pub fn ignore_notfound<T>(result: crate::io::Result<T>) -> crate::io::Result<()> {
    match result {
        Err(err) if err.kind() == crate::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Ok(()),
        Err(err) => Err(err),
    }
}
