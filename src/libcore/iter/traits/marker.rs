/// An iterator that always continues to yield `None` when exhausted.
///
/// Calling next on a fused iterator that has returned `None` once is guaranteed
/// to return [`None`] again. This trait should be implemented by all iterators
/// that behave this way because it allows optimizing [`Iterator::fuse`].
///
/// Note: In general, you should not use `FusedIterator` in generic bounds if
/// you need a fused iterator. Instead, you should just call [`Iterator::fuse`]
/// on the iterator. If the iterator is already fused, the additional [`Fuse`]
/// wrapper will be a no-op with no performance penalty.
///
/// [`None`]: ../../std/option/enum.Option.html#variant.None
/// [`Iterator::fuse`]: ../../std/iter/trait.Iterator.html#method.fuse
/// [`Fuse`]: ../../std/iter/struct.Fuse.html
#[stable(feature = "fused", since = "1.26.0")]
pub trait FusedIterator: Iterator {}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator + ?Sized> FusedIterator for &mut I {}

/// An iterator that reports an accurate length using size_hint.
///
/// The iterator reports a size hint where it is either exact
/// (lower bound is equal to upper bound), or the upper bound is [`None`].
/// The upper bound must only be [`None`] if the actual iterator length is
/// larger than [`usize::MAX`]. In that case, the lower bound must be
/// [`usize::MAX`], resulting in a [`.size_hint`] of `(usize::MAX, None)`.
///
/// The iterator must produce exactly the number of elements it reported
/// or diverge before reaching the end.
///
/// # Safety
///
/// This trait must only be implemented when the contract is upheld.
/// Consumers of this trait must inspect [`.size_hint`]’s upper bound.
///
/// [`None`]: ../../std/option/enum.Option.html#variant.None
/// [`usize::MAX`]: ../../std/usize/constant.MAX.html
/// [`.size_hint`]: ../../std/iter/trait.Iterator.html#method.size_hint
#[unstable(feature = "trusted_len", issue = "37572")]
pub unsafe trait TrustedLen : Iterator {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I: TrustedLen + ?Sized> TrustedLen for &mut I {}
