use core::alloc::{Layout, LayoutError};
use core::marker::PhantomData;
use core::mem::SizedTypeProperties;
use core::ptr::NonNull;

/// Calculates the offset of the value contained in a reference counted allocation.
///
/// Essentially, this calculates `size_of::<H>().next_multiple_of(value_or_allocation_alignment)`.
///
/// Note that `value_or_allocation_alignment` can be either the value alignment or the allocation
/// alignment (the greater one of the header alignment and the value alignment) because:
///
/// - If the value alignment is smaller than the header alignment, the result is `size_of::<H>()`,
///   which both choices produces.
/// - Otherwise the allocation alignment is exactly the value alignment.
///
/// # Safety
///
/// - `value_or_allocation_alignment` is a power of 2.
/// - The result does not overflow.
pub(super) const unsafe fn value_offset<H>(value_or_allocation_alignment: usize) -> usize {
    // SAFETY: Caller guarantees the result does not overflow.
    unsafe {
        let align_m1 = value_or_allocation_alignment.unchecked_sub(1);

        H::SIZE.unchecked_add(align_m1) & !align_m1
    }
}

/// A `Layout` that describes a reference-counted allocation containing a header of type `H`.
pub(in crate::rcs) struct RcLayout<H>(Layout, PhantomData<H>);

impl<H> RcLayout<H> {
    /// Tries to create an `RcLayout` to store a value with layout `value_layout`. Returns `Err` if
    /// `value_layout` is too big to store in a reference-counted allocation.
    #[inline]
    pub(in crate::rcs) const fn try_from_value_layout(
        value_layout: Layout,
    ) -> Result<Self, LayoutError> {
        match H::LAYOUT.extend(value_layout) {
            Ok((rc_layout, _)) => Ok(Self(rc_layout, PhantomData)),
            Err(error) => Err(error),
        }
    }

    /// Creates an `RcLayout` to store a value with layout `value_layout`.
    ///
    /// # Safety
    ///
    /// `RcLayout::try_from_value_layout(value_layout)` must return `Ok`.
    #[inline]
    pub(in crate::rcs) const unsafe fn from_value_layout_unchecked(value_layout: Layout) -> Self {
        // SAFETY: Caller guarantees the layout create will not fail.
        unsafe {
            let value_align = value_layout.align();
            let value_offset = value_offset::<H>(value_align);
            let size = value_offset.unchecked_add(value_layout.size());
            let align = if H::ALIGN < value_align { value_align } else { H::ALIGN };

            Self(Layout::from_size_align_unchecked(size, align), PhantomData)
        }
    }

    /// Creates an `RcLayout` to store a value with layout `value_layout`. Panics if `value_layout`
    /// is too big to store in a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(in crate::rcs) fn from_value_layout(value_layout: Layout) -> Self {
        Self::try_from_value_layout(value_layout).expect("capacity overflow")
    }

    /// Creates an `RcLayout` for storing a value that is pointed to by `value_ptr`.
    ///
    /// # Safety
    ///
    /// `value_ptr` must have correct metadata for `T`.
    #[cfg(not(no_global_oom_handling))]
    pub(in crate::rcs) unsafe fn from_value_ptr<T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is `Sized`,
        /// the `RcLayout` value is computed at compile time.
        trait SpecRcLayout<H> {
            unsafe fn spec_rc_layout(value_ptr: NonNull<Self>) -> RcLayout<H>;
        }

        impl<H, T> SpecRcLayout<H> for T
        where
            T: ?Sized,
        {
            #[inline]
            default unsafe fn spec_rc_layout(value_ptr: NonNull<Self>) -> RcLayout<H> {
                // SAFETY: Caller guarantees the validify of `value_ptr`.
                RcLayout::from_value_layout(unsafe { Layout::for_value_raw(value_ptr.as_ptr()) })
            }
        }

        impl<H, T> SpecRcLayout<H> for T {
            #[inline]
            unsafe fn spec_rc_layout(_: NonNull<Self>) -> RcLayout<H> {
                Self::RC_LAYOUT
            }
        }

        // SAFETY: Caller guarantees the validify of `value_ptr`.
        unsafe { T::spec_rc_layout(value_ptr) }
    }

    /// Creates an `RcLayout` for storing a value that is pointed to by `value_ptr`, assuming the
    /// value is small enough to fit inside a reference-counted allocation.
    ///
    /// # Safety
    ///
    /// - `value_ptr` must have correct metadata for a `T` object.
    /// - It must be known that the memory layout described by `value_ptr` can be used to create an
    ///   `RcLayout` successfully.
    pub(in crate::rcs) unsafe fn from_value_ptr_unchecked<T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is `Sized`,
        /// the `RcLayout` value is computed at compile time.
        trait SpecRcLayoutUnchecked<H> {
            unsafe fn spec_rc_layout_unchecked(value_ptr: NonNull<Self>) -> RcLayout<H>;
        }

        impl<H, T> SpecRcLayoutUnchecked<H> for T
        where
            T: ?Sized,
        {
            #[inline]
            default unsafe fn spec_rc_layout_unchecked(value_ptr: NonNull<Self>) -> RcLayout<H> {
                // SAFETY: Caller guarantees the validify of `value_ptr`.
                unsafe {
                    RcLayout::from_value_layout_unchecked(Layout::for_value_raw(value_ptr.as_ptr()))
                }
            }
        }

        impl<H, T> SpecRcLayoutUnchecked<H> for T {
            #[inline]
            unsafe fn spec_rc_layout_unchecked(_: NonNull<Self>) -> RcLayout<H> {
                Self::RC_LAYOUT
            }
        }

        // SAFETY: Caller guarantees the validify of `value_ptr`.
        unsafe { T::spec_rc_layout_unchecked(value_ptr) }
    }

    /// Creates an `RcLayout` for storing `value` that is pointed to by `value_ptr`, assuming the
    /// value is small enough to fit inside a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(in crate::rcs) fn from_value<T>(value: &T) -> Self
    where
        T: ?Sized,
    {
        // SAFETY: Pointers derived from references is guaranteed to be valid.
        unsafe { Self::from_value_ptr(NonNull::from_ref(value)) }
    }

    /// Creates an `RcLayout` to store an array of `length` elements of type `T`. Panics if the
    /// array is too big to store in a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    pub(in crate::rcs) fn new_array<T>(length: usize) -> Self {
        T::LAYOUT
            .repeat_packed(length)
            .and_then(RcLayout::try_from_value_layout)
            .expect("capacity overflow")
    }

    /// Returns an `Layout` object that describes the reference-counted allocation.
    pub(in crate::rcs) const fn get(&self) -> Layout {
        self.0
    }

    /// Returns the byte offset of the value stored in a reference-counted allocation that is
    /// described by `self`.
    #[inline]
    pub(in crate::rcs) const fn value_offset(&self) -> usize {
        // SAFETY: Invariant of `self` guarantees `self.0` describes a valid reference counted
        // allocation.
        unsafe { value_offset::<H>(self.0.align()) }
    }

    /// Returns the byte size of the value stored in a reference-counted allocation that is
    /// described by `self`.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(in crate::rcs) const fn value_size(&self) -> usize {
        // SAFETY: `value_offset()` is guaranteed to be less than `size()` since we have reference
        // counters before value.
        unsafe { self.0.size().unchecked_sub(self.value_offset()) }
    }
}

impl<H> Clone for RcLayout<H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H> Copy for RcLayout<H> {}

pub(in crate::rcs) trait RcLayoutExt<H> {
    /// Computes `RcLayout` at compile time if `Self` is `Sized`.
    const RC_LAYOUT: RcLayout<H>;
}

impl<H, T> RcLayoutExt<H> for T {
    const RC_LAYOUT: RcLayout<H> = match RcLayout::try_from_value_layout(T::LAYOUT) {
        Ok(rc_layout) => rc_layout,
        Err(_) => panic!("value is too big to store in a reference-counted allocation"),
    };
}
