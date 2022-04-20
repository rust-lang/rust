/// An iterator whose items are random-accessible efficiently
///
/// # Safety
///
/// The iterator's `size_hint` must be exact and cheap to call.
///
/// `TrustedRandomAccess::size` may not be overridden.
///
/// All subtypes and all supertypes of `Self` must also implement `TrustedRandomAccess`.
/// In particular, this means that types with non-invariant parameters usually can not have
/// an impl for `TrustedRandomAccess` that depends on any trait bounds on such parameters, except
/// for bounds that come from the respective struct/enum definition itself, or bounds involving
/// traits that themselves come with a guarantee similar to this one.
///
/// If `Self: ExactSizeIterator` then `self.len()` must always produce results consistent
/// with `self.size()`.
///
/// If `Self: Iterator`, then `<Self as Iterator>::__iterator_get_unchecked(&mut self, idx)`
/// must be safe to call provided the following conditions are met.
///
/// 1. `0 <= idx` and `idx < self.size()`.
/// 2. If `Self: !Clone`, then `self.__iterator_get_unchecked(idx)` is never called with the same
///    index on `self` more than once.
/// 3. After `self.__iterator_get_unchecked(idx)` has been called, then `self.next_back()` will
///    only be called at most `self.size() - idx - 1` times. If `Self: Clone` and `self` is cloned,
///    then this number is calculated for `self` and its clone individually,
///    but `self.next_back()` calls that happened before the cloning count for both `self` and the clone.
/// 4. After `self.__iterator_get_unchecked(idx)` has been called, then only the following methods
///    will be called on `self` or on any new clones of `self`:
///     * `std::clone::Clone::clone`
///     * `std::iter::Iterator::size_hint`
///     * `std::iter::DoubleEndedIterator::next_back`
///     * `std::iter::ExactSizeIterator::len`
///     * `std::iter::Iterator::__iterator_get_unchecked`
///     * `std::iter::TrustedRandomAccess::size`
///
/// Further, given that these conditions are met, it must guarantee that:
///
/// * It does not change the value returned from `size_hint`
/// * It must be safe to call the methods listed above on `self` after calling
///   `self.__iterator_get_unchecked(idx)`, assuming that the required traits are implemented.
/// * It must also be safe to drop `self` after calling `self.__iterator_get_unchecked(idx)`.
//
// FIXME: Clarify interaction with SourceIter/InPlaceIterable. Calling `SourceIter::as_inner`
// after `__iterator_get_unchecked` is supposed to be allowed.
#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
#[rustc_specialization_trait]
pub unsafe trait TrustedRandomAccess: Sized {
    // Convenience method.
    fn size(&self) -> usize
    where
        Self: Iterator,
    {
        self.size_hint().0
    }

    fn cleanup(&mut self, num: usize, forward: bool);
}

// The following marker traits exist because specializing on them currently is the only way to avoid
// emitting dead IR. Associated constants do not work because we currently don't have post-monomorphization
// DCE.
//
// Pulling in the setup and cleanup methods on every specialized `for _ in` loop leads to 10% IR bloat
// and LLVM won't eliminate it in debug mode.

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
#[rustc_specialization_trait]
#[marker]
pub unsafe trait TrustedRandomAccessNeedsCleanup {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
#[rustc_specialization_trait]
#[marker]
pub unsafe trait TrustedRandomAccessNeedsForwardSetup {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
#[rustc_specialization_trait]
#[marker]
pub unsafe trait TrustedRandomAccessNeedsReverseSetup {}

/// Like `Iterator::__iterator_get_unchecked`, but doesn't require the compiler to
/// know that `U: TrustedRandomAccess`.
///
/// ## Safety
///
/// Same requirements calling `get_unchecked` directly.
#[doc(hidden)]
#[inline]
pub(in crate::iter) unsafe fn try_get_unchecked<I>(it: &mut I, idx: usize) -> I::Item
where
    I: Iterator,
{
    // SAFETY: the caller must uphold the contract for
    // `Iterator::__iterator_get_unchecked`.
    unsafe { it.try_get_unchecked(idx) }
}

unsafe trait SpecTrustedRandomAccess: Iterator {
    /// If `Self: TrustedRandomAccess`, it must be safe to call
    /// `Iterator::__iterator_get_unchecked(self, index)`.
    unsafe fn try_get_unchecked(&mut self, index: usize) -> Self::Item;
}

unsafe impl<I: Iterator> SpecTrustedRandomAccess for I {
    default unsafe fn try_get_unchecked(&mut self, _: usize) -> Self::Item {
        panic!("Should only be called on TrustedRandomAccess iterators");
    }
}

unsafe impl<I: Iterator + TrustedRandomAccess> SpecTrustedRandomAccess for I {
    #[inline]
    unsafe fn try_get_unchecked(&mut self, index: usize) -> Self::Item {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { self.__iterator_get_unchecked(index) }
    }
}
