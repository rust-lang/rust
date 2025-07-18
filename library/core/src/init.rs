//! In-place initialization.
//!
//! This module describes the interface through which types supporting in-place initialization
//! can perform initialization with minimal or zero additional allocations or moves.

use crate::alloc::Layout;
use crate::ptr::{Pointee, Unique, from_raw_parts_mut};

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
/// ``` ignore (illustrative)
/// use std::alloc::alloc;
/// fn init_boxed<T: ?Sized + Pointee, I: Init<T>>(init: I) -> Result<Box<T>, I::Error> {
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
    /// This method must return a layout that precisely matches
    /// with `T`.
    /// Namely the size and the alignment must be equal.
    #[lang = "init_layout"]
    fn layout(&self) -> Layout;

    /// Writes a valid value of type `T` to `slot` or fails.
    ///
    /// If this call returns [`Ok`], then `slot` is guaranteed to contain a valid
    /// value of type `T`.
    /// Otherwise, in case the result is an [`Err`], the implementation must guarantee
    /// that the `slot` may be repurposed for future reuse.
    ///
    /// If `T` is unsized, then `slot` may be combined with
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
    /// The caller must also make sure that there is no possible aliasing
    /// to the slot that the caller is supplying.
    #[lang = "init_init"]
    unsafe fn init(self, slot: *mut ()) -> Result<T::Metadata, Self::Error>;
}

// There is no intention to stabilise this contraption.
// It shall remain internal.
#[doc(hidden)]
#[unstable(feature = "in_place_initialization_guard", issue = "999999")]
#[lang = "init_guard"]
#[allow(missing_debug_implementations)]
pub struct InitGuard<T: ?Sized + Pointee>(Unique<T>);

/// InitGuard is a guard to track the live state of initialised sub-fields of `struct`s,
/// tuples and fix-length array elements.
///
/// ```ignore (illustrative)
/// init MyStruct {
///     a: init1,
///     b: init2,
/// }
/// // is conceptually lowered to
/// let slot_MyStruct: *mut ();
/// let slot_a: *mut ();
/// let slot_b: *mut ();
/// let () = init1.init(slot_a)?;
/// let guard_a = InitGuard::new(slot_a, ());
/// let metadata = init2.init(slot_b)?;
/// let guard_b = InitGuard::new(slot_b, metadata);
/// // [ accepted state ]
/// guard_a.release();
/// guard_b.release();
/// // [ commit MyStruct ]
/// Ok(metadata)
/// ```
impl<T: ?Sized + Pointee> InitGuard<T> {
    /// # Safety
    ///
    /// The caller in built MIR must make sure that the result of the in-place
    /// initialization is actually a success before constructing a guard.
    #[inline(always)]
    #[lang = "init_guard_new"]
    unsafe fn new(slot: *mut (), metadata: <T as Pointee>::Metadata) -> Self {
        // SAFETY: the precondition of this call means that the slot is fully initialized
        // and the caller is the owner of the data.
        // Any aliasing cannot happen.
        unsafe { Self(Unique::new_unchecked(from_raw_parts_mut(slot, metadata))) }
    }

    /// # Safety
    ///
    /// The caller from MIR built from `init` blocks must make sure that
    /// this function is called only when all in-place initialization
    /// in the target slot is complete and it is ready to commit the place
    /// into the accepted state.
    #[inline(always)]
    #[lang = "init_guard_release"]
    unsafe fn release(self) {
        crate::mem::forget(self);
    }
}

unsafe impl<#[may_dangle] T: ?Sized + Pointee> Drop for InitGuard<T> {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: given the precondition to the constructor of the guard,
        // the ownership of the slot is unique and the slot has been fully
        // initialized.
        unsafe {
            if Layout::for_value_raw(self.0.as_ptr()).size() > 0 {
                crate::ptr::drop_in_place(self.0.as_ptr());
            }
        }
    }
}

// There is no intention to stabilise this contraption.
#[doc(hidden)]
#[unstable(feature = "in_place_initialization_guard", issue = "999999")]
#[lang = "init_array_guard"]
#[allow(missing_debug_implementations)]
// NOTE: even though we know that here `T: Sized` but the trait solver
// cannot conclude that `T: Pointee<Metadata = ()>`.
// Remove it once we sort this one out in the solver.
pub struct InitArrayGuard<T: Pointee<Metadata = ()>>(Unique<T>, usize);

/// InitArrayGuard is used to track the progress of constructing repeating
/// in-place initialization for `[{init}; <count>]`
///
/// ```ignore (illustrative)
/// init [some_init; 4]
/// // to be lowered into
/// let mut slot: *mut ();
/// let mut guard = InitArrayGuard::new()
/// let mut count = 0;
/// let mut layout: Layout = some_init.layout();
/// loop {
///     if count >= 4 { break; }
///     some_init.init(slot);
///     slot = (slot as *mut T).wrapping_add(1) as *mut ();
///     guard.bump();
/// }
/// // [ accepted state ]
/// guard.release();
/// // [ commit [some_init; 4] ]
/// Ok(4)
/// ```
impl<T: Pointee<Metadata = ()>> InitArrayGuard<T> {
    /// # Safety
    ///
    /// The built MIR assumes that Init::init precondition is upheld,
    /// so that the `slot` is not null, with exact memory allocation
    /// so that `T` fits perfectly with correct alignment.
    ///
    /// The `Init` protocal also implies that there is no aliasing on any part of
    /// the supplied slot.
    #[inline(always)]
    #[lang = "init_array_guard_new"]
    unsafe fn new(slot: *mut ()) -> Self {
        // SAFETY: the precondition of this call means that the slot is fully initialized
        // and the caller is the owner of the data.
        // Any aliasing cannot happen.
        unsafe { Self(Unique::new_unchecked(from_raw_parts_mut(slot, ())), 0) }
    }

    /// # Safety
    ///
    /// The caller from MIR built from `init` blocks must make sure that
    /// this function is called only when the result from a `Init::init` call
    /// with the last vacant slot is actually successful.
    ///
    /// The caller MIR also must make sure that the total number of calls to
    /// this function never exceeds the limit that the supplied slot allows.
    #[inline(always)]
    #[lang = "init_array_guard_bump"]
    unsafe fn bump(&mut self) {
        self.1 += 1;
    }

    /// # Safety
    ///
    /// The caller from MIR built from `init` blocks must make sure that
    /// this function is called only when all in-place initialization
    /// in the target slot is complete and it is ready to commit the place
    /// into the accepted state and hand the ownership off.
    #[inline(always)]
    #[lang = "init_array_guard_release"]
    unsafe fn release(self) {
        core::mem::forget(self);
    }
}

unsafe impl<#[may_dangle] T: Pointee<Metadata = ()>> Drop for InitArrayGuard<T> {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: the precondition to construction of this guard and calls to
        // `bump` ensures that the constructed sub-array stays in the bound and no
        // aliasing is possible on this array.
        unsafe {
            let (ptr, ()) = self.0.as_ptr().to_raw_parts();
            if Layout::for_value_raw(ptr).size() > 0 && self.1 > 0 {
                let to_drop: *mut [T] = from_raw_parts_mut(ptr, self.1);
                core::ptr::drop_in_place(to_drop);
            }
        }
    }
}
