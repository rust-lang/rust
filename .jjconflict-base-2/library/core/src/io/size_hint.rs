/// Internal trait used to allow for specialization in `Read::size_hint` and
/// `Iterator::size_hint`.
///
/// All types implement this through a blanket default implementation returning
/// a hint of `(0, None)`.
///
/// Implementors should only provide [`lower_bound`](SizeHint::lower_bound) and
/// [`upper_bound`](SizeHint::upper_bound).
/// [`size_hint`](SizeHint::size_hint) is provided as a `final` method to enforce
/// correctness.
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub trait SizeHint {
    /// Returns a lower bound on the number of elements this container-like item
    /// contains.
    /// For example, an array `[u8; 12]` could return any value between `0` and
    /// `12` inclusively as a correct implementation.
    ///
    /// Through specialization, all types implement this method returning a default
    /// value of `0`.
    ///
    /// Implementations *must* ensure the returned value is less than or equal to
    /// the true element count.
    fn lower_bound(&self) -> usize;

    /// Returns an upper bound on the number of elements this container-like item
    /// contains if it can be determined, otherwise `None`.
    ///
    /// Through specialization, all types implement this method returning a default
    /// value of `None`.
    ///
    /// Implementations *must* ensure the returned value is greater than or equal
    /// to the true element count.
    fn upper_bound(&self) -> Option<usize>;

    /// Returns an estimate for the number of elements this container like type
    /// contains.
    ///
    /// This is a `final` method, and is guaranteed to return
    /// `(self.lower_bound(), self.upper_bound())`.
    ///
    /// Without specialization, types implementing this trait will return `(0, None)`.
    final fn size_hint(&self) -> (usize, Option<usize>) {
        (self.lower_bound(), self.upper_bound())
    }
}

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
impl<T: ?Sized> SizeHint for T {
    #[inline]
    default fn lower_bound(&self) -> usize {
        0
    }

    #[inline]
    default fn upper_bound(&self) -> Option<usize> {
        None
    }
}
