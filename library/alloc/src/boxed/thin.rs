// Based on
// https://github.com/matthieu-m/rfc2580/blob/b58d1d3cba0d4b5e859d3617ea2d0943aaa31329/examples/thin.rs
// by matthieu-m
use crate::alloc::{self, Layout, LayoutError};
use core::fmt::{self, Debug, Display, Formatter};
use core::marker::{PhantomData, Unsize};
use core::mem;
use core::ops::{Deref, DerefMut};
use core::ptr::Pointee;
use core::ptr::{self, NonNull};

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
    ptr: WithHeader<<T as Pointee>::Metadata>,
    _marker: PhantomData<T>,
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T> ThinBox<T> {
    /// Moves a type to the heap with its `Metadata` stored in the heap allocation instead of on
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
    #[cfg(not(no_global_oom_handling))]
    pub fn new(value: T) -> Self {
        let meta = ptr::metadata(&value);
        let ptr = WithHeader::new(meta, value);
        ThinBox { ptr, _marker: PhantomData }
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<Dyn: ?Sized> ThinBox<Dyn> {
    /// Moves a type to the heap with its `Metadata` stored in the heap allocation instead of on
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
    #[cfg(not(no_global_oom_handling))]
    pub fn new_unsize<T>(value: T) -> Self
    where
        T: Unsize<Dyn>,
    {
        let meta = ptr::metadata(&value as &Dyn);
        let ptr = WithHeader::new(meta, value);
        ThinBox { ptr, _marker: PhantomData }
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
            self.ptr.drop::<T>(value);
        }
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized> ThinBox<T> {
    fn meta(&self) -> <T as Pointee>::Metadata {
        //  Safety:
        //  -   NonNull and valid.
        unsafe { *self.ptr.header() }
    }

    fn data(&self) -> *mut u8 {
        self.ptr.value()
    }
}

/// A pointer to type-erased data, guaranteed to have a header `H` before the pointed-to location.
struct WithHeader<H>(NonNull<u8>, PhantomData<H>);

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
            let ptr = alloc::alloc(layout);

            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            //  Safety:
            //  -   The size is at least `aligned_header_size`.
            let ptr = ptr.add(value_offset) as *mut _;

            let ptr = NonNull::new_unchecked(ptr);

            let result = WithHeader(ptr, PhantomData);
            ptr::write(result.header(), header);
            ptr::write(result.value().cast(), value);

            result
        }
    }

    //  Safety:
    //  -   Assumes that `value` can be dereferenced.
    unsafe fn drop<T: ?Sized>(&self, value: *mut T) {
        unsafe {
            // SAFETY: Layout must have been computable if we're in drop
            let (layout, value_offset) =
                Self::alloc_layout(Layout::for_value_raw(value)).unwrap_unchecked();

            ptr::drop_in_place::<T>(value);
            // We only drop the value because the Pointee trait requires that the metadata is copy
            // aka trivially droppable
            alloc::dealloc(self.0.as_ptr().sub(value_offset), layout);
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
        unsafe { self.0.as_ptr().sub(Self::header_size()) as *mut H }
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
