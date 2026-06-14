use core::alloc::{Allocator, Layout};
use core::mem;
use core::ptr::{self, NonNull, Pointee};

#[cfg(not(no_global_oom_handling))]
use crate::alloc::handle_alloc_error;
use crate::boxed::Box;

pub(crate) struct UninitBox<T: Pointee + ?Sized, A: Allocator> {
    ptr: NonNull<T>,
    alloc: A,
}

impl<T: Pointee + ?Sized, A: Allocator> UninitBox<T, A> {
    /// # Safety
    ///
    /// `meta` must hold the same conditions as for [`Layout::for_value_raw`].
    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn new_for_metadata_in(meta: T::Metadata, alloc: A) -> Self {
        let ptr = ptr::from_raw_parts_mut::<T>(ptr::null_mut::<()>(), meta);
        // SAFETY: guaranteed by caller
        let layout = unsafe { Layout::for_value_raw(ptr) };

        let ptr = if layout.size() == 0 {
            layout.dangling().cast::<()>()
        } else {
            alloc.allocate(layout).unwrap_or_else(|_| handle_alloc_error(layout)).cast()
        };
        let ptr = NonNull::from_raw_parts(ptr, meta);

        Self { ptr, alloc }
    }

    #[inline]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub(crate) fn into_raw_with_allocator(b: Self) -> (NonNull<T>, A) {
        let b = mem::ManuallyDrop::new(b);
        let alloc = unsafe { ptr::read(&b.alloc) };
        (b.ptr, alloc)
    }

    /// # Safety
    ///
    /// The inner value must be initialized properly.
    #[inline]
    pub(crate) unsafe fn assume_init(self) -> Box<T, A> {
        let (ptr, alloc) = Self::into_raw_with_allocator(self);
        unsafe { Box::from_non_null_in(ptr, alloc) }
    }
}

impl<T: Pointee + ?Sized, A: Allocator> Drop for UninitBox<T, A> {
    #[inline]
    fn drop(&mut self) {
        // Safety: by invariant of this type, `ptr` is a valid pointer to an allocation
        unsafe {
            let layout = Layout::for_value_raw(self.ptr.as_ptr());
            if layout.size() != 0 {
                self.alloc.deallocate(self.ptr.cast(), layout);
            }
        }
    }
}
