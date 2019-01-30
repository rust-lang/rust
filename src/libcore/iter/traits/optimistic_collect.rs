/// A specialized trait designed to improve the estimates used when preallocating collections in
/// cases where `size_hint` is too conservative. For instance, when collecting into an `Option` or a
/// `Result`, the most common outcome is a non-empty collection, but the protocol allows `size_hint`
/// to only provide a lower bound of `0`. `OptimisticCollect` can be specialized for such cases in
/// order to optimize the creation of the resulting collections without breaking `Iterator` rules.
#[unstable(feature = "optimistic_collect", issue = "00000")]
pub trait OptimisticCollect: Iterator {
    /// Provides an estimate of the size of the iterator for the purposes of preallocating
    /// collections that can be built from it. By default it provides the lower bound of
    /// `size_hint`.
    fn optimistic_collect_count(&self) -> usize;
}

#[unstable(feature = "optimistic_collect", issue = "00000")]
impl<I: Iterator> OptimisticCollect for I {
    default fn optimistic_collect_count(&self) -> usize { self.size_hint().0 }
}

#[unstable(feature = "optimistic_collect", issue = "00000")]
impl<I: Iterator> OptimisticCollect for &mut I {
    default fn optimistic_collect_count(&self) -> usize { (**self).size_hint().0 }
}
