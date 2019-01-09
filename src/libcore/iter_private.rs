/// An iterator whose items are random accessible efficiently
///
/// # Safety
///
/// The iterator's .len() and size_hint() must be exact.
/// `.len()` must be cheap to call.
///
/// .get_unchecked() must return distinct mutable references for distinct
/// indices (if applicable), and must return a valid reference if index is in
/// 0..self.len().
#[doc(hidden)]
pub unsafe trait TrustedRandomAccess : ExactSizeIterator {
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item;
    /// Returns `true` if getting an iterator element may have
    /// side effects. Remember to take inner iterators into account.
    fn may_have_side_effect() -> bool;
}
