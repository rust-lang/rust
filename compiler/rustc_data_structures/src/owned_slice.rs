//! A type that stores a boxed type that derefs to a slice, and a custom index into that slice.
//!
//! The type can be mapped to have a different index, allowing you to take subslices while still
//! preserving the original type.
//!
//! This is a specialized version of `owning_ref`, because `owning_ref` has various soundness
//! issues that arise from its generality, that this one does not have.

// FIXME(nilstrieb): This could probably use some more trait implementations,
// though they are currently not required in the compiler.

use std::marker::PhantomData;
use std::ops::Deref;

/// An owned subslice of an owned value.
/// The owned value must be behind a [`Deref`] indirection, for example a [`Box`] or an [`Lrc`].
///
/// The `OwnWrap` is an indirection that derefs to the inner owner (`Own`). The inner owner
/// then derefs to the actual slice, of which this type references a subslice.
///
/// Can be further subsliced using [`Self::map`].
///
/// [`Lrc`]: crate::sync::Lrc
pub struct OwnedSlice<OwnWrap, T> {
    /// The wrapper around the owned value. Derefs to the owned value, which then derefs to the slice.
    owned: OwnWrap,
    /// The start value of the subslice.
    start: usize,
    /// The length of the subslice.
    len: usize,
    /// +--------------------------+
    /// | We conceptually own a T. |
    /// +----+  +------------------+
    ///       \/
    /// boo! ⊂(´･◡･⊂ )∘˚˳°
    _boo: PhantomData<T>,
}

impl<OwnWrap, Own: ?Sized, T> OwnedSlice<OwnWrap, T>
where
    OwnWrap: Deref<Target = Own>,
    Own: Deref<Target = [T]> + 'static,
{
    /// Create a new `OwnedSlice<OwnWrap, T>`. Sets the subslice to contain the full slice that `OwnWrap`
    /// nestedly derefs to.
    pub fn new(owned: OwnWrap) -> Self {
        let len = owned.len();
        Self { owned, start: 0, len, _boo: PhantomData }
    }

    /// Change the slice to a smaller subslice, while retaining ownership over the full value.
    ///
    /// # Panics
    /// Panics if the subslice is out of bounds of the smaller subslice.
    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(&[T]) -> &[T],
    {
        self.try_map::<_, !>(|slice| Ok(f(slice))).unwrap()
    }

    /// Map the slice to a subslice, while retaining ownership over the full value.
    /// The function may be fallible.
    ///
    /// # Panics
    /// Panics if the returned subslice is out of bounds of the base slice.
    pub fn try_map<F, E>(self, f: F) -> Result<Self, E>
    where
        F: FnOnce(&[T]) -> Result<&[T], E>,
    {
        let base_slice = self.base_slice();
        let std::ops::Range { start: base_ptr, end: base_end_ptr } = base_slice.as_ptr_range();
        let base_len = base_slice.len();

        let slice = &base_slice[self.start..][..self.len];
        let slice = f(slice)?;
        let (slice_ptr, len) = (slice.as_ptr(), slice.len());

        let start = if len == 0 {
            // For empty slices, we don't really care where the start is. Also, the start of the subslice could
            // be a one-past-the-end pointer, which we cannot allow in the code below, but is ok here.
            0
        } else {
            // Assert that the start pointer is in bounds, I.E. points to the same allocated object.
            // If the slice is empty or contains a zero-sized type, the start and end pointers of the
            // base slice will always be the same, meaning this check will always fail.
            assert!(base_ptr <= slice_ptr);
            assert!(slice_ptr < base_end_ptr);

            // SAFETY: We have checked that the `slice_ptr` is bigger than the `base_ptr`.
            // We have also checked that it's in bounds of the allocated object.
            let diff_in_bytes = unsafe { slice_ptr.cast::<u8>().sub_ptr(base_ptr.cast::<u8>()) };

            // The subslice might not actually be a difference of sizeof(T), but truncating it should be fine.
            diff_in_bytes / std::mem::size_of::<T>()
        };

        // Assert that the length is not out of bounds. This is not nessecary for soundness, but helps detect errors
        // early, instead of panicking in the deref.
        assert!((start + len) <= base_len);

        Ok(Self { owned: self.owned, start, len, _boo: PhantomData })
    }

    fn base_slice(&self) -> &[T] {
        &*self.owned
    }
}

impl<OwnWrap, Own: ?Sized, T> Deref for OwnedSlice<OwnWrap, T>
where
    OwnWrap: Deref<Target = Own>,
    Own: Deref<Target = [T]> + 'static,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        let base_slice = self.base_slice();
        &base_slice[self.start..][..self.len]
    }
}

impl<OwnWrap, Own: ?Sized, T> std::borrow::Borrow<[T]> for OwnedSlice<OwnWrap, T>
where
    OwnWrap: Deref<Target = Own>,
    Own: Deref<Target = [T]> + 'static,
{
    fn borrow(&self) -> &[T] {
        &*self
    }
}

#[cfg(test)]
mod tests;
