use super::Vec;
use crate::alloc::{Allocator, Global};
use crate::raw_vec::RawVec;
use core::fmt;
use core::iter::{FusedIterator, InPlaceIterable, SourceIter, TrustedLen, TrustedRandomAccess};
use core::marker::PhantomData;
use core::mem::{self, ManuallyDrop};
use core::ops::{Range, Try};
use core::ptr::{self, NonNull};
use core::slice::{self};

/// An iterator that moves out of a vector.
///
/// This `struct` is created by the `into_iter` method on [`Vec`](super::Vec)
/// (provided by the [`IntoIterator`] trait).
///
/// # Example
///
/// ```
/// let v = vec![0, 1, 2];
/// let iter: std::vec::IntoIter<_> = v.into_iter();
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    pub(super) buf: NonNull<T>,
    pub(super) phantom: PhantomData<T>,
    pub(super) cap: usize,
    pub(super) alloc: A,
    pub(super) alive: Range<usize>,
}

#[stable(feature = "vec_intoiter_debug", since = "1.13.0")]
impl<T: fmt::Debug, A: Allocator> fmt::Debug for IntoIter<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

impl<T, A: Allocator> IntoIter<T, A> {
    /// Returns the remaining items of this iterator as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = vec!['a', 'b', 'c'];
    /// let mut into_iter = vec.into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// let _ = into_iter.next().unwrap();
    /// assert_eq!(into_iter.as_slice(), &['b', 'c']);
    /// ```
    #[stable(feature = "vec_into_iter_as_slice", since = "1.15.0")]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.buf.as_ptr().offset(self.alive.start as isize), self.len())
        }
    }

    /// Returns the remaining items of this iterator as a mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// let vec = vec!['a', 'b', 'c'];
    /// let mut into_iter = vec.into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// into_iter.as_mut_slice()[2] = 'z';
    /// assert_eq!(into_iter.next().unwrap(), 'a');
    /// assert_eq!(into_iter.next().unwrap(), 'b');
    /// assert_eq!(into_iter.next().unwrap(), 'z');
    /// ```
    #[stable(feature = "vec_into_iter_as_slice", since = "1.15.0")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { &mut *self.as_raw_mut_slice() }
    }

    /// Returns a reference to the underlying allocator.
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns a pointer offset by `self.alive.start`
    fn as_mut(&self) -> *mut T {
        unsafe { self.buf.as_ptr().offset(self.alive.start as isize) }
    }

    fn as_raw_mut_slice(&mut self) -> *mut [T] {
        ptr::slice_from_raw_parts_mut(self.as_mut(), self.len())
    }

    /// Drops remaining elements and relinquishes the backing allocation.
    ///
    /// This is roughly equivalent to the following, but more efficient
    ///
    /// ```
    /// # let mut into_iter = Vec::<u8>::with_capacity(10).into_iter();
    /// (&mut into_iter).for_each(core::mem::drop);
    /// unsafe { core::ptr::write(&mut into_iter, Vec::new().into_iter()); }
    /// ```
    #[cfg(not(no_global_oom_handling))]
    pub(super) fn forget_allocation_drop_remaining(&mut self) {
        let remaining = self.as_raw_mut_slice();

        // overwrite the individual fields instead of creating a new
        // struct and then overwriting &mut self.
        // this creates less assembly
        self.cap = 0;
        self.buf = unsafe { NonNull::new_unchecked(RawVec::NEW.ptr()) };
        self.alive.start = self.alive.end;

        unsafe {
            ptr::drop_in_place(remaining);
        }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(super) fn to_vec(self) -> Vec<T, A> {
        unsafe {
            let this = ManuallyDrop::new(self);
            if this.alive.start > 0 {
                ptr::copy(this.as_mut(), this.buf.as_ptr(), this.len());
            }
            Vec::from_raw_parts_in(this.buf.as_ptr(), this.len(), this.cap, ptr::read(&this.alloc))
        }
    }
}

#[stable(feature = "vec_intoiter_as_ref", since = "1.46.0")]
impl<T, A: Allocator> AsRef<[T]> for IntoIter<T, A> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send, A: Allocator + Send> Send for IntoIter<T, A> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync, A: Allocator> Sync for IntoIter<T, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.alive.next().map(|idx| unsafe {
            if mem::size_of::<T>() == 0 {
                mem::zeroed()
            } else {
                ptr::read(self.buf.as_ptr().offset(idx as isize))
            }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.alive.len();
        (exact, Some(exact))
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        let buf = self.buf;
        self.alive.try_fold(init, |acc, idx| {
            let val = unsafe { ptr::read(buf.as_ptr().offset(idx as isize)) };
            f(acc, val)
        })
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    unsafe fn __iterator_get_unchecked(&mut self, i: usize) -> Self::Item
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: the caller must guarantee that `i` is in bounds of the
        // `Vec<T>`, so `i` cannot overflow an `isize`, and the `self.ptr.add(i)`
        // is guaranteed to pointer to an element of the `Vec<T>` and
        // thus guaranteed to be valid to dereference.
        //
        // Also note the implementation of `Self: TrustedRandomAccess` requires
        // that `T: Copy` so reading elements from the buffer doesn't invalidate
        // them for `Drop`.
        unsafe {
            if mem::size_of::<T>() == 0 {
                mem::zeroed()
            } else {
                ptr::read(self.buf.as_ptr().offset(self.alive.start as isize).add(i))
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator> DoubleEndedIterator for IntoIter<T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.alive.next_back().map(|idx| unsafe {
            if mem::size_of::<T>() == 0 {
                mem::zeroed()
            } else {
                ptr::read(self.buf.as_ptr().offset(idx as isize))
            }
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: Allocator> ExactSizeIterator for IntoIter<T, A> {
    fn is_empty(&self) -> bool {
        self.alive.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, A: Allocator> FusedIterator for IntoIter<T, A> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T, A: Allocator> TrustedLen for IntoIter<T, A> {}

#[doc(hidden)]
#[unstable(issue = "none", feature = "std_internals")]
// T: Copy as approximation for !Drop since get_unchecked does not advance self.ptr
// and thus we can't implement drop-handling
unsafe impl<T, A: Allocator> TrustedRandomAccess for IntoIter<T, A>
where
    T: Copy,
{
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "vec_into_iter_clone", since = "1.8.0")]
impl<T: Clone, A: Allocator + Clone> Clone for IntoIter<T, A> {
    #[cfg(not(test))]
    fn clone(&self) -> Self {
        self.as_slice().to_vec_in(self.alloc.clone()).into_iter()
    }
    #[cfg(test)]
    fn clone(&self) -> Self {
        crate::slice::to_vec(self.as_slice(), self.alloc.clone()).into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T, A: Allocator> Drop for IntoIter<T, A> {
    fn drop(&mut self) {
        struct DropGuard<'a, T, A: Allocator>(&'a mut IntoIter<T, A>);

        impl<T, A: Allocator> Drop for DropGuard<'_, T, A> {
            fn drop(&mut self) {
                unsafe {
                    // `IntoIter::alloc` is not used anymore after this
                    let alloc = ptr::read(&self.0.alloc);
                    // RawVec handles deallocation
                    let _ = RawVec::from_raw_parts_in(self.0.buf.as_ptr(), self.0.cap, alloc);
                }
            }
        }

        let guard = DropGuard(self);
        // destroy the remaining elements
        unsafe {
            ptr::drop_in_place(guard.0.as_raw_mut_slice());
        }
        // now `guard` will be dropped and do the rest
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T, A: Allocator> InPlaceIterable for IntoIter<T, A> {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<T, A: Allocator> SourceIter for IntoIter<T, A> {
    type Source = Self;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut Self::Source {
        self
    }
}

// internal helper trait for in-place iteration specialization.
#[rustc_specialization_trait]
pub(crate) trait AsIntoIter {
    type Item;
    fn as_into_iter(&mut self) -> &mut IntoIter<Self::Item>;
}

impl<T> AsIntoIter for IntoIter<T> {
    type Item = T;

    fn as_into_iter(&mut self) -> &mut IntoIter<Self::Item> {
        self
    }
}
