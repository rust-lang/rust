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
