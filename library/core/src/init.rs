//! In-place initialization
//!
//! This module describes the interface through which types supporting in-place initialization
//! interact with allocation mechanism to ensure correct and safe initialization of values
//! within the memory slots provided by the allocation.

use crate::ptr::Pointee;

/// # Safety
///
/// Implementers must ensure that if `init` returns `Ok(metadata)`, then
/// `core::ptr::from_raw_parts_mut(slot, metadata)` must reference a valid
/// value owned by the caller. Furthermore, the layout returned by using
/// `size_of` and `align_of` on this pointer must match what `Self::layout()`
/// returns exactly.
///
/// Implementers must ensure that the implementation of `init()` does not rely
/// on the value being pinned.
#[unstable(feature = "in_place_initialization", issue = "999999")]
#[lang = "init_trait"]
pub unsafe trait Init<T: ?Sized + Pointee> {
    /// Error type upon initialization failure during the actual
    /// in-place initialization procedure.
    #[lang = "init_error"]
    type Error;

    /// The layout needed by this initializer, which the allocation
    /// should arrange the destination memory slot accordingly.
    #[lang = "init_layout"]
    fn layout(&self) -> crate::alloc::Layout;

    /// Writes a valid value of type `T` to `slot` or fails.
    ///
    /// If this call returns [`Ok`], then `slot` is guaranteed to contain a valid
    /// value of type `T`. If `T` is unsized, then `slot` may be combined with
    /// the metadata to obtain a valid pointer to the value.
    ///
    /// Note that `slot` should be thought of as a `*mut T`. A unit type is used
    /// so that the pointer is thin even if `T` is unsized.
    ///
    /// # Safety
    ///
    /// The caller must provide a pointer that references a location that `init`
    /// may write to, and the location must have at least the size and alignment
    /// specified by [`Init::layout`].
    ///
    /// If this call returns `Ok` and the initializer does not implement
    /// `Init<T>`, then `slot` contains a pinned value, and the caller must
    /// respect the usual pinning requirements for `slot`.
    #[lang = "init_init"]
    unsafe fn init(self, slot: *mut ()) -> Result<T::Metadata, Self::Error>;
}
