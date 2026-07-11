use core::alloc::{Layout, LayoutError};
use core::mem::SizedTypeProperties;
use core::ptr::NonNull;

use crate::raw_rc::RefCounts;

/// A `Layout` that describes a reference-counted allocation.
#[derive(Clone, Copy)]
pub(crate) struct RcLayout(Layout);

impl RcLayout {
    /// Tries to create an `RcLayout` to store a value with layout `value_layout`. Returns `Err` if
    /// `value_layout` is too big to store in a reference-counted allocation.
    #[inline]
    pub(crate) const fn try_from_value_layout(value_layout: Layout) -> Result<Self, LayoutError> {
        match RefCounts::LAYOUT.extend(value_layout) {
            Ok((rc_layout, _)) => Ok(Self(rc_layout)),
            Err(error) => Err(error),
        }
    }

    /// Tries to create an `RcLayout` to store a value with layout `value_layout`. Returns `Err` if
    /// `value_layout` is too big to store in a reference-counted allocation.
    #[inline]
    pub(crate) const fn try_from_value<T>(value: &T) -> Result<Self, LayoutError>
    where
        T: ?Sized,
    {
        Self::try_from_value_layout(Layout::for_value(value))
    }

    /// Creates an `RcLayout` to store a value with layout `value_layout`.
    ///
    /// # Safety
    ///
    /// `RcLayout::try_from_value_layout(value_layout)` must return `Ok`.
    #[inline]
    pub(crate) unsafe fn from_value_layout_unchecked(value_layout: Layout) -> Self {
        unsafe { Self::try_from_value_layout(value_layout).unwrap_unchecked() }
    }

    /// Creates an `RcLayout` to store a value with layout `value_layout`. Panics if `value_layout`
    /// is too big to store in a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(crate) fn from_value_layout(value_layout: Layout) -> Self {
        Self::try_from_value_layout(value_layout).unwrap()
    }

    /// Creates an `RcLayout` for storing a value that is pointed to by `value_ptr`.
    ///
    /// # Safety
    ///
    /// `value_ptr` must have correct metadata for `T`.
    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn from_value_ptr<T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is `Sized`,
        /// the `RcLayout` value is computed at compile time.
        trait SpecRcLayout {
            unsafe fn spec_rc_layout(value_ptr: NonNull<Self>) -> RcLayout;
        }

        impl<T> SpecRcLayout for T
        where
            T: ?Sized,
        {
            #[inline]
            default unsafe fn spec_rc_layout(value_ptr: NonNull<Self>) -> RcLayout {
                RcLayout::from_value_layout(unsafe { Layout::for_value_raw(value_ptr.as_ptr()) })
            }
        }

        impl<T> SpecRcLayout for T {
            #[inline]
            unsafe fn spec_rc_layout(_: NonNull<Self>) -> RcLayout {
                Self::RC_LAYOUT
            }
        }

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
    pub(crate) unsafe fn from_value_ptr_unchecked<T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is `Sized`,
        /// the `RcLayout` value is computed at compile time.
        trait SpecRcLayoutUnchecked {
            unsafe fn spec_rc_layout_unchecked(value_ptr: NonNull<Self>) -> RcLayout;
        }

        impl<T> SpecRcLayoutUnchecked for T
        where
            T: ?Sized,
        {
            #[inline]
            default unsafe fn spec_rc_layout_unchecked(value_ptr: NonNull<Self>) -> RcLayout {
                unsafe {
                    RcLayout::from_value_layout_unchecked(Layout::for_value_raw(value_ptr.as_ptr()))
                }
            }
        }

        impl<T> SpecRcLayoutUnchecked for T {
            #[inline]
            unsafe fn spec_rc_layout_unchecked(_: NonNull<Self>) -> RcLayout {
                Self::RC_LAYOUT
            }
        }

        unsafe { T::spec_rc_layout_unchecked(value_ptr) }
    }

    /// Creates an `RcLayout` for storing `value` that is pointed to by `value_ptr`, assuming the
    /// value is small enough to fit inside a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(crate) fn from_value<T>(value: &T) -> Self
    where
        T: ?Sized,
    {
        unsafe { Self::from_value_ptr(NonNull::from_ref(value)) }
    }

    /// Creates an `RcLayout` to store an array of `length` elements of type `T`. Panics if the
    /// array is too big to store in a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_array<T>(length: usize) -> Self {
        /// For minimizing monomorphization cost.
        #[inline]
        fn inner(value_layout: Layout, length: usize) -> RcLayout {
            // We can use `repeat_packed` here because the outer function passes `T::LAYOUT` as the
            // `value_layout`, which is already padded to a multiple of its alignment.
            value_layout.repeat_packed(length).and_then(RcLayout::try_from_value_layout).unwrap()
        }

        inner(T::LAYOUT, length)
    }

    /// Returns an `Layout` object that describes the reference-counted allocation.
    pub(crate) const fn get(&self) -> Layout {
        self.0
    }

    /// Returns the byte offset of the value stored in a reference-counted allocation that is
    /// described by `self`.
    #[inline]
    pub(crate) const fn value_offset(&self) -> usize {
        // SAFETY:
        //
        // This essentially calculates `size_of::<RefCounts>().next_multiple_of(self.align())`.
        //
        // See the comments in `Layout::size_rounded_up_to_custom_align` for a detailed explanation.
        unsafe {
            let align_m1 = self.0.align().unchecked_sub(1);

            size_of::<RefCounts>().unchecked_add(align_m1) & !align_m1
        }
    }

    /// Returns the byte size of the value stored in a reference-counted allocation that is
    /// described by `self`.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(crate) const fn value_size(&self) -> usize {
        unsafe { self.0.size().unchecked_sub(self.value_offset()) }
    }
}

pub(crate) trait RcLayoutExt {
    /// Computes `RcLayout` at compile time if `Self` is `Sized`.
    const RC_LAYOUT: RcLayout;
}

impl<T> RcLayoutExt for T {
    const RC_LAYOUT: RcLayout = match RcLayout::try_from_value_layout(T::LAYOUT) {
        Ok(rc_layout) => rc_layout,
        Err(_) => panic!("value is too big to store in a reference-counted allocation"),
    };
}
