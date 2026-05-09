//! Traits for vectors of pointers.

mod const_ptr;
mod mut_ptr;

mod sealed {
    pub trait Sealed {}
}

pub use const_ptr::*;
pub use mut_ptr::*;

use crate::simd::Simd;

/// Creates pointers with the given addresses and no provenance.
///
/// Equivalent to calling [`core::ptr::without_provenance`] on each element.
#[inline]
pub fn without_provenance<T, const N: usize>(addr: Simd<usize, N>) -> Simd<*const T, N> {
    // An int-to-pointer transmute currently has exactly the intended semantics: it creates a
    // pointer without provenance. Note that this is *not* a stable guarantee about transmute
    // semantics, it relies on sysroot crates having special status.
    // SAFETY: every valid integer is also a valid pointer (as long as you don't dereference that
    // pointer).
    unsafe { core::mem::transmute_copy(&addr) }
}

/// Creates mutable pointers with the given addresses and no provenance.
///
/// Equivalent to calling [`core::ptr::without_provenance_mut`] on each element.
#[inline]
pub fn without_provenance_mut<T, const N: usize>(addr: Simd<usize, N>) -> Simd<*mut T, N> {
    // An int-to-pointer transmute currently has exactly the intended semantics: it creates a
    // pointer without provenance. Note that this is *not* a stable guarantee about transmute
    // semantics, it relies on sysroot crates having special status.
    // SAFETY: every valid integer is also a valid pointer (as long as you don't dereference that
    // pointer).
    unsafe { core::mem::transmute_copy(&addr) }
}

/// Converts addresses back to pointers, picking up some previously "exposed" provenance.
///
/// Equivalent to calling [`core::ptr::with_exposed_provenance`] on each element.
#[inline]
pub fn with_exposed_provenance<T, const N: usize>(addr: Simd<usize, N>) -> Simd<*const T, N> {
    // SAFETY: addr is a vector of usize
    unsafe { core::intrinsics::simd::simd_with_exposed_provenance(addr) }
}

/// Converts addresses back to mutable pointers, picking up some previously "exposed" provenance.
///
/// Equivalent to calling [`core::ptr::with_exposed_provenance_mut`] on each element.
#[inline]
pub fn with_exposed_provenance_mut<T, const N: usize>(addr: Simd<usize, N>) -> Simd<*mut T, N> {
    // SAFETY: addr is a vector of usize
    unsafe { core::intrinsics::simd::simd_with_exposed_provenance(addr) }
}
