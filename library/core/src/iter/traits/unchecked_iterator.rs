use crate::iter::TrustedLen;

/// [`TrustedLen`] cannot have methods, so this allows augmenting it.
///
/// It currently requires `TrustedLen` because it's unclear whether it's
/// reasonably possible to depend on the `size_hint` of anything else.
pub(crate) trait UncheckedIterator: TrustedLen {
    /// Gets the next item from a non-empty iterator.
    ///
    /// Because there's always a value to return, that means it can return
    /// the `Item` type directly, without wrapping it in an `Option`.
    ///
    /// # Safety
    ///
    /// This can only be called if `size_hint().0 != 0`, guaranteeing that
    /// there's at least one item available.
    ///
    /// Otherwise (aka when `size_hint().1 == Some(0)`), this is UB.
    ///
    /// # Note to Implementers
    ///
    /// This has a default implementation using [`Option::unwrap_unchecked`].
    /// That's probably sufficient if your `next` *always* returns `Some`,
    /// such as for infinite iterators.  In more complicated situations, however,
    /// sometimes there can still be `insertvalue`/`assume`/`extractvalue`
    /// instructions remaining in the IR from the `Option` handling, at which
    /// point you might want to implement this manually instead.
    #[unstable(feature = "trusted_len_next_unchecked", issue = "37572")]
    #[inline]
    unsafe fn next_unchecked(&mut self) -> Self::Item {
        let opt = self.next();
        // SAFETY: The caller promised that we're not empty, and
        // `Self: TrustedLen` so we can actually trust the `size_hint`.
        unsafe { opt.unwrap_unchecked() }
    }
}

/// Specialization trait for unchecked iterator indexing
///
/// # Safety and Use
///
/// * `size_hint` must be exact
/// * the size must fit into an `usize`
/// * each position must only be accessed once
/// * no other iterator methods must be called between getting the size,
///   calling an unchecked getter one or more times and
///   finally comitting the index updates with an unchecked setter.
/// * changing iteration direction requires comitting the index updates
///   from the previous iteration direction
///
/// The start-end range represents inclusive-exclusive bounds.
/// Meaning that the end index 0 represents a positon one-past the current
/// range of valid items.
/// While start index 0 represents the first item if the iterator is non-empty.
///
///
#[doc(hidden)]
#[unstable(feature = "trusted_indexed_access", issue = "none")]
#[rustc_specialization_trait]
pub trait UncheckedIndexedIterator: Sized {
    /// `true` if getting an iterator element may have side effects.
    /// Remember to take inner iterators into account.
    /// `true` is the conservative choice.
    const MAY_HAVE_SIDE_EFFECT: bool;

    /// `true` if updating interator indexes is needed
    /// even if the iterator will no longer be used, e.g. to avoid double-drops or leaks.
    /// `trus` is the conservative choice.
    const CLEANUP_ON_DROP: bool;

    unsafe fn set_front_index_from_end_unchecked(&mut self, _new_len: usize, _old_len: usize);
    unsafe fn set_end_index_from_start_unchecked(&mut self, _new_len: usize, _old_len: usize);
}

pub(in crate::iter) unsafe trait SpecIndexedAccess: Iterator {
    /// If `Self: TrustedRandomAccess`, it must be safe to call
    /// `Iterator::__iterator_get_unchecked(self, index)`.
    unsafe fn index_from_end_unchecked_inner(&mut self, index: usize) -> Self::Item;

    unsafe fn index_from_start_unchecked_inner(&mut self, index: usize) -> Self::Item;
}

unsafe impl<I: Iterator> SpecIndexedAccess for I {
    default unsafe fn index_from_end_unchecked_inner(&mut self, _: usize) -> Self::Item {
        unreachable!("Should only be called on UncheckedIndexedIterator iterators");
    }

    default unsafe fn index_from_start_unchecked_inner(&mut self, _: usize) -> Self::Item {
        unreachable!("Should only be called on UncheckedIndexedIterator iterators");
    }
}

unsafe impl<I: Iterator + UncheckedIndexedIterator> SpecIndexedAccess for I {
    #[inline]
    unsafe fn index_from_end_unchecked_inner(&mut self, idx: usize) -> Self::Item {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { self.index_from_end_unchecked(idx) }
    }

    #[inline]
    unsafe fn index_from_start_unchecked_inner(&mut self, idx: usize) -> Self::Item {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { self.index_from_start_unchecked(idx) }
    }
}
