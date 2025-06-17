//! In-place initialization.
//!
//! This module describes the interface through which types supporting in-place initialization
//! can perform initialization with minimal or zero additional allocations or moves.

use crate::ptr::Pointee;

/// In-place Initializer for `T`.
///
/// An instance of `Init<T>` carries all the information necessary to initialize a `T` in a
/// well-defined memory location, criteria of which is prescribed in the Safety section.
///
/// # Fallibility
///
/// The initialization might fail and return an error of type [`Self::Error`] instead.
/// In that case, the memory provided to [`Self::init`] shall be repurposed in any way,
/// even though it might have been written to by this initializer.
///
/// # Examples
///
/// ## Initializing unsized types
///
/// To initialize an unsized type, first query the required layout for `T` using [`Self::layout`].
/// Then provide a pointer to an allocation of at least the specified alignment and size.
///
/// If initialization was successful, then [`Self::init`] returns the metadata that combined with
/// the pointer to the given to [`Self::init`] yields a valid pointer to `T`.
///
/// ```
/// use std::alloc::alloc;
/// fn init_unsized<T: ?Sized + Pointee, I: Init<T>>(init: I) -> Result<Box<T>, I::Error> {
///     let layout = init.layout();
///     let memory = alloc(layout).cast::<()>();
///     let meta = init.init(memory)?;
///     Box::from_raw(from_raw_parts_mut(memory, meta))
/// }
/// ```
///
/// # Safety
///
/// Implementers must ensure that if [`self.init(slot)`] returns `Ok(metadata)`,
/// then [`core::ptr::from_raw_parts_mut(slot, metadata)`] must reference a valid
/// value owned by the caller.
/// Furthermore, the layout returned by using
/// [`core::intrinsics::size_of_val`] and [`core::intrinsics::align_of_val`] on this pointer
/// must match what [`Self::layout()`] returns exactly.
///
/// If `T` is sized, or in other words `T: Sized`, [`Init::layout`] in this case *must*
/// return the same layout as [`Layout::new::<T>()`] would.
///
/// Implementers must ensure that the implementation of `init()` does not rely
/// on the value being pinned.
///
/// [`core::ptr::from_raw_parts_mut(slot, metadata)`]: core::ptr::from_raw_parts_mut
/// [`Self::layout()`]: Init::layout
/// [`self.init(slot)`]: Init::init
/// [`Layout::new::<T>()`]: core::alloc::Layout::new
#[unstable(feature = "in_place_initialization", issue = "999999")]
#[lang = "init_trait"]
pub unsafe trait Init<T: ?Sized + Pointee> {
    /// Error type upon initialization failure during the actual
    /// in-place initialization procedure.
    #[lang = "init_error"]
    type Error;

    /// The layout needed by this initializer.
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
    #[lang = "init_init"]
    unsafe fn init(self, slot: *mut ()) -> Result<T::Metadata, Self::Error>;
}
