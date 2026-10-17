use core::alloc::{Layout, LayoutError};
use core::mem::SizedTypeProperties;
use core::ptr::NonNull;

/// A `Layout` that describes a reference-counted allocation containing a header of type `H`.
pub(super) struct RcLayout {
    layout: Layout,
    value_offset: usize,
}

impl RcLayout {
    /// Tries to create an `RcLayout` to store value. Returns `Err` if the value is too big to store
    /// in a reference-counted allocation.
    #[inline]
    pub(super) fn try_from_value<H, T>(value: &T) -> Result<Self, LayoutError>
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is `Sized`,
        /// the `RcLayout` value is computed at compile time.
        trait SpecTryRcLayout {
            fn spec_try_rc_layout<H>(&self) -> Result<RcLayout, LayoutError>;
        }

        impl<T> SpecTryRcLayout for T
        where
            T: ?Sized,
        {
            #[inline]
            default fn spec_try_rc_layout<H>(&self) -> Result<RcLayout, LayoutError> {
                RcLayout::try_from_value_layout::<H>(Layout::for_value(self))
            }
        }

        impl<T> SpecTryRcLayout for T {
            #[inline]
            fn spec_try_rc_layout<H>(&self) -> Result<RcLayout, LayoutError> {
                // Do we fail the compilation if we know the calculation overflows at compile time?
                const { RcLayout::try_from_value_layout::<H>(T::LAYOUT) }
            }
        }

        T::spec_try_rc_layout::<H>(value)
    }

    /// Tries to create an `RcLayout` to store a value with layout `value_layout`. Returns `Err` if
    /// `value_layout` is too big to store in a reference-counted allocation.
    #[inline]
    pub(super) const fn try_from_value_layout<H>(
        value_layout: Layout,
    ) -> Result<Self, LayoutError> {
        match H::LAYOUT.extend(value_layout) {
            Ok((layout, value_offset)) => Ok(Self { layout, value_offset }),
            Err(error) => Err(error),
        }
    }

    pub(super) const fn new<H, T>() -> Self {
        const {
            match Self::try_from_value_layout::<H>(T::LAYOUT) {
                Ok(rc_layout) => rc_layout,
                Err(_) => panic!("value is too big to store in a reference-counted allocation"),
            }
        }
    }

    /// Creates an `RcLayout` for storing `value` that is pointed to by `value_ptr`, assuming the
    /// value is small enough to fit inside a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) fn from_value<H, T>(value: &T) -> Self
    where
        T: ?Sized,
    {
        // SAFETY: Pointers derived from references is guaranteed to be valid.
        unsafe { Self::from_value_ptr::<H, T>(NonNull::from_ref(value)) }
    }

    /// Creates an `RcLayout` for storing `value`, assuming the value is small enough to fit inside
    /// a reference-counted allocation.
    ///
    /// # Safety
    ///
    /// Caller guarantees `T` we can store a `T` value inside a reference counted allocation with
    /// type `H`.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) unsafe fn from_value_unchecked<H, T>(value: &T) -> Self
    where
        T: ?Sized,
    {
        // SAFETY: Caller guarantees we are safe to calculate the allocation layout.
        unsafe { Self::from_value_ptr_unchecked::<H, T>(NonNull::from_ref(value)) }
    }

    /// Creates an `RcLayout` for storing a value that is pointed to by `value_ptr`.
    ///
    /// # Safety
    ///
    /// `value_ptr` must have correct metadata for `T`.
    #[cfg(not(no_global_oom_handling))]
    pub(super) unsafe fn from_value_ptr<H, T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is `Sized`,
        /// the `RcLayout` value is computed at compile time.
        trait SpecRcLayout {
            unsafe fn spec_rc_layout<H>(value_ptr: NonNull<Self>) -> RcLayout;
        }

        impl<T> SpecRcLayout for T
        where
            T: ?Sized,
        {
            #[inline]
            default unsafe fn spec_rc_layout<H>(value_ptr: NonNull<Self>) -> RcLayout {
                // SAFETY: Caller guarantees the validity of `value_ptr`.
                RcLayout::from_value_layout::<H>(unsafe {
                    Layout::for_value_raw(value_ptr.as_ptr())
                })
            }
        }

        impl<T> SpecRcLayout for T {
            #[inline]
            unsafe fn spec_rc_layout<H>(_: NonNull<Self>) -> RcLayout {
                const { RcLayout::new::<H, Self>() }
            }
        }

        // SAFETY: Caller guarantees the validity of `value_ptr`.
        unsafe { T::spec_rc_layout::<H>(value_ptr) }
    }

    /// Creates an `RcLayout` for storing a value that is pointed to by `value_ptr`, assuming the
    /// value is small enough to fit inside a reference-counted allocation.
    ///
    /// # Safety
    ///
    /// - `value_ptr` must have correct metadata for a `T` object.
    /// - It must be known that the memory layout described by `value_ptr` can be used to create an
    ///   `RcLayout` successfully.
    pub(super) unsafe fn from_value_ptr_unchecked<H, T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is `Sized`,
        /// the `RcLayout` value is computed at compile time.
        trait SpecTryRcLayoutUnchecked {
            unsafe fn spec_rc_layout_unchecked<H>(value_ptr: NonNull<Self>) -> RcLayout;
        }

        impl<T> SpecTryRcLayoutUnchecked for T
        where
            T: ?Sized,
        {
            #[inline]
            default unsafe fn spec_rc_layout_unchecked<H>(value_ptr: NonNull<Self>) -> RcLayout {
                // SAFETY: Caller guarantees the validity of `value_ptr`.
                unsafe {
                    RcLayout::from_value_layout_unchecked::<H>(Layout::for_value_raw(
                        value_ptr.as_ptr(),
                    ))
                }
            }
        }

        impl<T> SpecTryRcLayoutUnchecked for T {
            #[inline]
            unsafe fn spec_rc_layout_unchecked<H>(_: NonNull<Self>) -> RcLayout {
                const { RcLayout::new::<H, Self>() }
            }
        }

        // SAFETY: Caller guarantees the validity of `value_ptr`.
        unsafe { T::spec_rc_layout_unchecked::<H>(value_ptr) }
    }

    /// Creates an `RcLayout` to store a value with layout `value_layout`. Panics if `value_layout`
    /// is too big to store in a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) fn from_value_layout<H>(value_layout: Layout) -> Self {
        Self::try_from_value_layout::<H>(value_layout).expect("capacity overflow")
    }

    /// Creates an `RcLayout` to store a value with layout `value_layout`.
    ///
    /// # Safety
    ///
    /// `RcLayout::try_from_value_layout(value_layout)` must return `Ok`.
    #[inline]
    pub(super) const unsafe fn from_value_layout_unchecked<H>(value_layout: Layout) -> Self {
        // SAFETY: Caller guarantees the layout create will not fail.
        unsafe {
            let value_align = value_layout.align();

            // Calculates `H::SIZE.next_multiple_of(value_align)` given `value_align` being a
            // power of 2 and the result does not overflow.
            let value_align_m1 = value_align.unchecked_sub(1);
            let value_offset = H::SIZE.unchecked_add(value_align_m1) & !value_align_m1;

            let size = value_offset.unchecked_add(value_layout.size());
            let align = if H::ALIGN < value_align { value_align } else { H::ALIGN };

            Self { layout: Layout::from_size_align_unchecked(size, align), value_offset }
        }
    }

    /// Creates an `RcLayout` to store an array of `length` elements of type `T`. Panics if the
    /// array is too big to store in a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    pub(super) fn new_array<H, T>(length: usize) -> Self {
        T::LAYOUT
            .repeat_packed(length)
            .and_then(RcLayout::try_from_value_layout::<H>)
            .expect("capacity overflow")
    }

    /// Returns an `Layout` object that describes the reference-counted allocation.
    pub(super) const fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns the byte offset of the value stored in a reference-counted allocation that is
    /// described by `self`.
    #[inline]
    pub(super) const fn value_offset(&self) -> usize {
        self.value_offset
    }

    /// Returns the byte size of the value stored in a reference-counted allocation that is
    /// described by `self`.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) const fn value_size(&self) -> usize {
        // SAFETY: `value_offset` is guaranteed to be less than `size()`.
        unsafe { self.layout.size().unchecked_sub(self.value_offset) }
    }
}

impl Clone for RcLayout {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for RcLayout {}
