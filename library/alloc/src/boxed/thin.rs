//! Based on
//! <https://github.com/matthieu-m/rfc2580/blob/b58d1d3cba0d4b5e859d3617ea2d0943aaa31329/examples/thin.rs>
//! by matthieu-m

use core::error::Error;
use core::fmt::{self, Debug, Display, Formatter};
#[cfg(not(no_global_oom_handling))]
use core::intrinsics::const_allocate;
use core::marker::PhantomData;
#[cfg(not(no_global_oom_handling))]
use core::marker::Unsize;
use core::mem;
#[cfg(not(no_global_oom_handling))]
use core::mem::SizedTypeProperties;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull, Pointee};

use crate::alloc::{self, Layout, LayoutError};

/// ThinBox.
///
/// A thin pointer for heap allocation, regardless of T.
///
/// # Examples
///
/// ```
/// #![feature(thin_box)]
/// use std::boxed::ThinBox;
///
/// let five = ThinBox::new(5);
/// let thin_slice = ThinBox::<[i32]>::new_unsize([1, 2, 3, 4]);
///
/// use std::mem::{size_of, size_of_val};
/// let size_of_ptr = size_of::<*const ()>();
/// assert_eq!(size_of_ptr, size_of_val(&five));
/// assert_eq!(size_of_ptr, size_of_val(&thin_slice));
/// ```
#[unstable(feature = "thin_box", issue = "92791")]
pub struct ThinBox<T: ?Sized> {
    // This is essentially `WithHeader<<T as Pointee>::Metadata>`,
    // but that would be invariant in `T`, and we want covariance.
    ptr: WithOpaqueHeader,
    _marker: PhantomData<T>,
}

/// `ThinBox<T>` is `Send` if `T` is `Send` because the data is owned.
#[unstable(feature = "thin_box", issue = "92791")]
unsafe impl<T: ?Sized + Send> Send for ThinBox<T> {}

/// `ThinBox<T>` is `Sync` if `T` is `Sync` because the data is owned.
#[unstable(feature = "thin_box", issue = "92791")]
unsafe impl<T: ?Sized + Sync> Sync for ThinBox<T> {}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T> ThinBox<T> {
    /// Moves a type to the heap with its [`Metadata`] stored in the heap allocation instead of on
    /// the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(thin_box)]
    /// use std::boxed::ThinBox;
    ///
    /// let five = ThinBox::new(5);
    /// ```
    ///
    /// [`Metadata`]: core::ptr::Pointee::Metadata
    #[cfg(not(no_global_oom_handling))]
    pub fn new(value: T) -> Self {
        let meta = ptr::metadata(&value);
        let ptr = WithOpaqueHeader::new(meta, value);
        ThinBox { ptr, _marker: PhantomData }
    }

    /// Moves a type to the heap with its [`Metadata`] stored in the heap allocation instead of on
    /// the stack. Returns an error if allocation fails, instead of aborting.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// #![feature(thin_box)]
    /// use std::boxed::ThinBox;
    ///
    /// let five = ThinBox::try_new(5)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    ///
    /// [`Metadata`]: core::ptr::Pointee::Metadata
    pub fn try_new(value: T) -> Result<Self, core::alloc::AllocError> {
        let meta = ptr::metadata(&value);
        WithOpaqueHeader::try_new(meta, value).map(|ptr| ThinBox { ptr, _marker: PhantomData })
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<Dyn: ?Sized> ThinBox<Dyn> {
    /// Moves a type to the heap with its [`Metadata`] stored in the heap allocation instead of on
    /// the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(thin_box)]
    /// use std::boxed::ThinBox;
    ///
    /// let thin_slice = ThinBox::<[i32]>::new_unsize([1, 2, 3, 4]);
    /// ```
    ///
    /// [`Metadata`]: core::ptr::Pointee::Metadata
    #[cfg(not(no_global_oom_handling))]
    pub fn new_unsize<T>(value: T) -> Self
    where
        T: Unsize<Dyn>,
    {
        if mem::size_of::<T>() == 0 {
            let ptr = WithOpaqueHeader::new_unsize_zst::<Dyn, T>(value);
            ThinBox { ptr, _marker: PhantomData }
        } else {
            let meta = ptr::metadata(&value as &Dyn);
            let ptr = WithOpaqueHeader::new(meta, value);
            ThinBox { ptr, _marker: PhantomData }
        }
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized + Debug> Debug for ThinBox<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.deref(), f)
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized + Display> Display for ThinBox<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self.deref(), f)
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized> Deref for ThinBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        let value = self.data();
        let metadata = self.meta();
        let pointer = ptr::from_raw_parts(value as *const (), metadata);
        unsafe { &*pointer }
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized> DerefMut for ThinBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        let value = self.data();
        let metadata = self.meta();
        let pointer = ptr::from_raw_parts_mut::<T>(value as *mut (), metadata);
        unsafe { &mut *pointer }
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized> Drop for ThinBox<T> {
    fn drop(&mut self) {
        unsafe {
            let value = self.deref_mut();
            let value = value as *mut T;
            self.with_header().drop::<T>(value);
        }
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized> ThinBox<T> {
    fn meta(&self) -> <T as Pointee>::Metadata {
        //  Safety:
        //  -   NonNull and valid.
        unsafe { *self.with_header().header() }
    }

    fn data(&self) -> *mut u8 {
        self.with_header().value()
    }

    fn with_header(&self) -> &WithHeader<<T as Pointee>::Metadata> {
        // SAFETY: both types are transparent to `NonNull<u8>`
        unsafe { &*((&raw const self.ptr) as *const WithHeader<_>) }
    }
}

/// A pointer to type-erased data, guaranteed to either be:
/// 1. `NonNull::dangling()`, in the case where both the pointee (`T`) and
///    metadata (`H`) are ZSTs.
/// 2. A pointer to a valid `T` that has a header `H` directly before the
///    pointed-to location.
#[repr(transparent)]
struct WithHeader<H>(NonNull<u8>, PhantomData<H>);

/// An opaque representation of `WithHeader<H>` to avoid the
/// projection invariance of `<T as Pointee>::Metadata`.
#[repr(transparent)]
struct WithOpaqueHeader(NonNull<u8>);

impl WithOpaqueHeader {
    #[cfg(not(no_global_oom_handling))]
    fn new<H, T>(header: H, value: T) -> Self {
        let ptr = WithHeader::new(header, value);
        Self(ptr.0)
    }

    #[cfg(not(no_global_oom_handling))]
    fn new_unsize_zst<Dyn, T>(value: T) -> Self
    where
        Dyn: ?Sized,
        T: Unsize<Dyn>,
    {
        let ptr = WithHeader::<<Dyn as Pointee>::Metadata>::new_unsize_zst::<Dyn, T>(value);
        Self(ptr.0)
    }

    fn try_new<H, T>(header: H, value: T) -> Result<Self, core::alloc::AllocError> {
        WithHeader::try_new(header, value).map(|ptr| Self(ptr.0))
    }
}

impl<H> WithHeader<H> {
    #[cfg(not(no_global_oom_handling))]
    fn new<T>(header: H, value: T) -> WithHeader<H> {
        let value_layout = Layout::new::<T>();
        let Ok((layout, value_offset)) = Self::alloc_layout(value_layout) else {
            // We pass an empty layout here because we do not know which layout caused the
            // arithmetic overflow in `Layout::extend` and `handle_alloc_error` takes `Layout` as
            // its argument rather than `Result<Layout, LayoutError>`, also this function has been
            // stable since 1.28 ._.
            //
            // On the other hand, look at this gorgeous turbofish!
            alloc::handle_alloc_error(Layout::new::<()>());
        };

        unsafe {
            // Note: It's UB to pass a layout with a zero size to `alloc::alloc`, so
            // we use `layout.dangling()` for this case, which should have a valid
            // alignment for both `T` and `H`.
            let ptr = if layout.size() == 0 {
                // Some paranoia checking, mostly so that the ThinBox tests are
                // more able to catch issues.
                debug_assert!(value_offset == 0 && T::IS_ZST && H::IS_ZST);
                layout.dangling()
            } else {
                let ptr = alloc::alloc(layout);
                if ptr.is_null() {
                    alloc::handle_alloc_error(layout);
                }
                // Safety:
                // - The size is at least `aligned_header_size`.
                let ptr = ptr.add(value_offset) as *mut _;

                NonNull::new_unchecked(ptr)
            };

            let result = WithHeader(ptr, PhantomData);
            ptr::write(result.header(), header);
            ptr::write(result.value().cast(), value);

            result
        }
    }

    /// Non-panicking version of `new`.
    /// Any error is returned as `Err(core::alloc::AllocError)`.
    fn try_new<T>(header: H, value: T) -> Result<WithHeader<H>, core::alloc::AllocError> {
        let value_layout = Layout::new::<T>();
        let Ok((layout, value_offset)) = Self::alloc_layout(value_layout) else {
            return Err(core::alloc::AllocError);
        };

        unsafe {
            // Note: It's UB to pass a layout with a zero size to `alloc::alloc`, so
            // we use `layout.dangling()` for this case, which should have a valid
            // alignment for both `T` and `H`.
            let ptr = if layout.size() == 0 {
                // Some paranoia checking, mostly so that the ThinBox tests are
                // more able to catch issues.
                debug_assert!(
                    value_offset == 0 && mem::size_of::<T>() == 0 && mem::size_of::<H>() == 0
                );
                layout.dangling()
            } else {
                let ptr = alloc::alloc(layout);
                if ptr.is_null() {
                    return Err(core::alloc::AllocError);
                }

                // Safety:
                // - The size is at least `aligned_header_size`.
                let ptr = ptr.add(value_offset) as *mut _;

                NonNull::new_unchecked(ptr)
            };

            let result = WithHeader(ptr, PhantomData);
            ptr::write(result.header(), header);
            ptr::write(result.value().cast(), value);

            Ok(result)
        }
    }

    // `Dyn` is `?Sized` type like `[u32]`, and `T` is ZST type like `[u32; 0]`.
    #[cfg(not(no_global_oom_handling))]
    fn new_unsize_zst<Dyn, T>(value: T) -> WithHeader<H>
    where
        Dyn: Pointee<Metadata = H> + ?Sized,
        T: Unsize<Dyn>,
    {
        assert!(mem::size_of::<T>() == 0);

        const fn max(a: usize, b: usize) -> usize {
            if a > b { a } else { b }
        }

        // Compute a pointer to the right metadata. This will point to the beginning
        // of the header, past the padding, so the assigned type makes sense.
        // It also ensures that the address at the end of the header is sufficiently
        // aligned for T.
        let alloc: &<Dyn as Pointee>::Metadata = const {
            // FIXME: just call `WithHeader::alloc_layout` with size reset to 0.
            // Currently that's blocked on `Layout::extend` not being `const fn`.

            let alloc_align =
                max(mem::align_of::<T>(), mem::align_of::<<Dyn as Pointee>::Metadata>());

            let alloc_size =
                max(mem::align_of::<T>(), mem::size_of::<<Dyn as Pointee>::Metadata>());

            unsafe {
                // SAFETY: align is power of two because it is the maximum of two alignments.
                let alloc: *mut u8 = const_allocate(alloc_size, alloc_align);

                let metadata_offset =
                    alloc_size.checked_sub(mem::size_of::<<Dyn as Pointee>::Metadata>()).unwrap();
                // SAFETY: adding offset within the allocation.
                let metadata_ptr: *mut <Dyn as Pointee>::Metadata =
                    alloc.add(metadata_offset).cast();
                // SAFETY: `*metadata_ptr` is within the allocation.
                metadata_ptr.write(ptr::metadata::<Dyn>(ptr::dangling::<T>() as *const Dyn));

                // SAFETY: we have just written the metadata.
                &*(metadata_ptr)
            }
        };

        // SAFETY: `alloc` points to `<Dyn as Pointee>::Metadata`, so addition stays in-bounds.
        let value_ptr =
            unsafe { (alloc as *const <Dyn as Pointee>::Metadata).add(1) }.cast::<T>().cast_mut();
        debug_assert!(value_ptr.is_aligned());
        mem::forget(value);
        WithHeader(NonNull::new(value_ptr.cast()).unwrap(), PhantomData)
    }

    // Safety:
    // - Assumes that either `value` can be dereferenced, or is the
    //   `NonNull::dangling()` we use when both `T` and `H` are ZSTs.
    unsafe fn drop<T: ?Sized>(&self, value: *mut T) {
        struct DropGuard<H> {
            ptr: NonNull<u8>,
            value_layout: Layout,
            _marker: PhantomData<H>,
        }

        impl<H> Drop for DropGuard<H> {
            fn drop(&mut self) {
                // All ZST are allocated statically.
                if self.value_layout.size() == 0 {
                    return;
                }

                unsafe {
                    // SAFETY: Layout must have been computable if we're in drop
                    let (layout, value_offset) =
                        WithHeader::<H>::alloc_layout(self.value_layout).unwrap_unchecked();

                    // Since we only allocate for non-ZSTs, the layout size cannot be zero.
                    debug_assert!(layout.size() != 0);
                    alloc::dealloc(self.ptr.as_ptr().sub(value_offset), layout);
                }
            }
        }

        unsafe {
            // `_guard` will deallocate the memory when dropped, even if `drop_in_place` unwinds.
            let _guard = DropGuard {
                ptr: self.0,
                value_layout: Layout::for_value_raw(value),
                _marker: PhantomData::<H>,
            };

            // We only drop the value because the Pointee trait requires that the metadata is copy
            // aka trivially droppable.
            ptr::drop_in_place::<T>(value);
        }
    }

    fn header(&self) -> *mut H {
        //  Safety:
        //  - At least `size_of::<H>()` bytes are allocated ahead of the pointer.
        //  - We know that H will be aligned because the middle pointer is aligned to the greater
        //    of the alignment of the header and the data and the header size includes the padding
        //    needed to align the header. Subtracting the header size from the aligned data pointer
        //    will always result in an aligned header pointer, it just may not point to the
        //    beginning of the allocation.
        let hp = unsafe { self.0.as_ptr().sub(Self::header_size()) as *mut H };
        debug_assert!(hp.is_aligned());
        hp
    }

    fn value(&self) -> *mut u8 {
        self.0.as_ptr()
    }

    const fn header_size() -> usize {
        mem::size_of::<H>()
    }

    fn alloc_layout(value_layout: Layout) -> Result<(Layout, usize), LayoutError> {
        Layout::new::<H>().extend(value_layout)
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized + Error> Error for ThinBox<T> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.deref().source()
    }
}
