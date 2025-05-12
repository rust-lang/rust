use crate::mem::{self, MaybeUninit};
use crate::ptr;

/// Private specialization trait used by CloneToUninit, as per
/// [the dev guide](https://std-dev-guide.rust-lang.org/policy/specialization.html).
pub(super) unsafe trait CopySpec: Clone {
    unsafe fn clone_one(src: &Self, dst: *mut Self);
    unsafe fn clone_slice(src: &[Self], dst: *mut [Self]);
}

unsafe impl<T: Clone> CopySpec for T {
    #[inline]
    default unsafe fn clone_one(src: &Self, dst: *mut Self) {
        // SAFETY: The safety conditions of clone_to_uninit() are a superset of those of
        // ptr::write().
        unsafe {
            // We hope the optimizer will figure out to create the cloned value in-place,
            // skipping ever storing it on the stack and the copy to the destination.
            ptr::write(dst, src.clone());
        }
    }

    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    default unsafe fn clone_slice(src: &[Self], dst: *mut [Self]) {
        let len = src.len();
        // This is the most likely mistake to make, so check it as a debug assertion.
        debug_assert_eq!(
            len,
            dst.len(),
            "clone_to_uninit() source and destination must have equal lengths",
        );

        // SAFETY: The produced `&mut` is valid because:
        // * The caller is obligated to provide a pointer which is valid for writes.
        // * All bytes pointed to are in MaybeUninit, so we don't care about the memory's
        //   initialization status.
        let uninit_ref = unsafe { &mut *(dst as *mut [MaybeUninit<T>]) };

        // Copy the elements
        let mut initializing = InitializingSlice::from_fully_uninit(uninit_ref);
        for element_ref in src {
            // If the clone() panics, `initializing` will take care of the cleanup.
            initializing.push(element_ref.clone());
        }
        // If we reach here, then the entire slice is initialized, and we've satisfied our
        // responsibilities to the caller. Disarm the cleanup guard by forgetting it.
        mem::forget(initializing);
    }
}

// Specialized implementation for types that are [`Copy`], not just [`Clone`],
// and can therefore be copied bitwise.
unsafe impl<T: Copy> CopySpec for T {
    #[inline]
    unsafe fn clone_one(src: &Self, dst: *mut Self) {
        // SAFETY: The safety conditions of clone_to_uninit() are a superset of those of
        // ptr::copy_nonoverlapping().
        unsafe {
            ptr::copy_nonoverlapping(src, dst, 1);
        }
    }

    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_slice(src: &[Self], dst: *mut [Self]) {
        let len = src.len();
        // This is the most likely mistake to make, so check it as a debug assertion.
        debug_assert_eq!(
            len,
            dst.len(),
            "clone_to_uninit() source and destination must have equal lengths",
        );

        // SAFETY: The safety conditions of clone_to_uninit() are a superset of those of
        // ptr::copy_nonoverlapping().
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), len);
        }
    }
}

/// Ownership of a collection of values stored in a non-owned `[MaybeUninit<T>]`, some of which
/// are not yet initialized. This is sort of like a `Vec` that doesn't own its allocation.
/// Its responsibility is to provide cleanup on unwind by dropping the values that *are*
/// initialized, unless disarmed by forgetting.
///
/// This is a helper for `impl<T: Clone> CloneToUninit for [T]`.
struct InitializingSlice<'a, T> {
    data: &'a mut [MaybeUninit<T>],
    /// Number of elements of `*self.data` that are initialized.
    initialized_len: usize,
}

impl<'a, T> InitializingSlice<'a, T> {
    #[inline]
    fn from_fully_uninit(data: &'a mut [MaybeUninit<T>]) -> Self {
        Self { data, initialized_len: 0 }
    }

    /// Push a value onto the end of the initialized part of the slice.
    ///
    /// # Panics
    ///
    /// Panics if the slice is already fully initialized.
    #[inline]
    fn push(&mut self, value: T) {
        MaybeUninit::write(&mut self.data[self.initialized_len], value);
        self.initialized_len += 1;
    }
}

impl<'a, T> Drop for InitializingSlice<'a, T> {
    #[cold] // will only be invoked on unwind
    fn drop(&mut self) {
        let initialized_slice = ptr::slice_from_raw_parts_mut(
            MaybeUninit::slice_as_mut_ptr(self.data),
            self.initialized_len,
        );
        // SAFETY:
        // * the pointer is valid because it was made from a mutable reference
        // * `initialized_len` counts the initialized elements as an invariant of this type,
        //   so each of the pointed-to elements is initialized and may be dropped.
        unsafe {
            ptr::drop_in_place::<[T]>(initialized_slice);
        }
    }
}
