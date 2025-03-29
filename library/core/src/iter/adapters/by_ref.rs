use crate::array;
use crate::iter::{FusedIterator, TrustedLen};
use crate::num::NonZero;
use crate::ops::{ChangeOutputType, NeverShortCircuit, Residual, Try};

/// Implements `Iterator` for mutable references to iterators, such as those produced by [`Iterator::by_ref`].
///
/// This implementation passes all method calls on to the original iterator.
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator + ?Sized> Iterator for &mut I {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        ByRef::<I>(self).next()
    }

    fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[Self::Item; N], array::IntoIter<Self::Item, N>> {
        ByRef::<I>(self).next_chunk()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        I::size_hint(self)
    }

    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        ByRef::<I>(self).advance_by(n)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        ByRef::<I>(self).nth(n)
    }

    fn for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    {
        ByRef::<I>(self).for_each(f)
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        ByRef::<I>(self).try_fold(init, f)
    }

    fn try_for_each<F, R>(&mut self, f: F) -> R
    where
        F: FnMut(Self::Item) -> R,
        R: Try<Output = ()>,
    {
        ByRef::<I>(self).try_for_each(f)
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        ByRef::<I>(self).fold(init, f)
    }

    fn reduce<F>(self, f: F) -> Option<Self::Item>
    where
        F: FnMut(Self::Item, Self::Item) -> Self::Item,
    {
        ByRef::<I>(self).reduce(f)
    }

    fn try_reduce<R>(
        &mut self,
        f: impl FnMut(Self::Item, Self::Item) -> R,
    ) -> ChangeOutputType<R, Option<R::Output>>
    where
        R: Try<Output = Self::Item, Residual: Residual<Option<Self::Item>>>,
    {
        ByRef::<I>(self).try_reduce(f)
    }

    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        ByRef::<I>(self).all(f)
    }

    fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        ByRef::<I>(self).any(f)
    }

    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        ByRef::<I>(self).find(predicate)
    }

    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        ByRef::<I>(self).find_map(f)
    }

    fn try_find<R>(
        &mut self,
        f: impl FnMut(&Self::Item) -> R,
    ) -> ChangeOutputType<R, Option<Self::Item>>
    where
        R: Try<Output = bool, Residual: Residual<Option<Self::Item>>>,
    {
        ByRef::<I>(self).try_find(f)
    }

    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        ByRef::<I>(self).position(predicate)
    }

    // FIXME: also specialize rposition
    //
    // Unfortunately, it's not possible to infer
    //
    // `I: DoubleEndedIterator + ExactSizeIterator`
    //
    // from
    //
    // `&mut I: DoubleEndedIterator + ExactSizeIterator`
    //
    // which makes it impossible to call the inner `rposition`, even with
    // specialization.
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for &mut I {
    fn next_back(&mut self) -> Option<Self::Item> {
        ByRef::<I>(self).next_back()
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        ByRef::<I>(self).advance_back_by(n)
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        ByRef::<I>(self).nth_back(n)
    }

    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        ByRef::<I>(self).try_rfold(init, f)
    }

    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        ByRef::<I>(self).rfold(init, f)
    }

    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        ByRef::<I>(self).rfind(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator + ?Sized> ExactSizeIterator for &mut I {
    fn len(&self) -> usize {
        I::len(self)
    }

    fn is_empty(&self) -> bool {
        I::is_empty(self)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator + ?Sized> FusedIterator for &mut I {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I: TrustedLen + ?Sized> TrustedLen for &mut I {}

// The following implementations use UFCS-style, rather than trusting autoderef,
// to avoid accidentally calling the `&mut Iterator` implementations.

/// A helper struct that implements all overridden `Iterator` methods by
/// forwarding them to either [`ByRefDefault`] or the underlying iterator.
///
/// It's `std`'s policy to not expose specialization publicly, hence we do not
/// specialize the public `Iterator for &mut I` implementation but only the
/// `Iterator` implementation of this private helper struct.
struct ByRef<'a, I: ?Sized>(&'a mut I);

// Make sure to add all iterator methods that can be forwarded in any way to
// a `Sized` underlying iterator here. Sometimes, e.g. in the case of `fold`
// that can involve going through the fallible version of a functions, since
// they mostly take a reference instead of consuming the iterator.
impl<I: Iterator + ?Sized> Iterator for ByRef<'_, I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        ByRefDefault::<I>(self.0).next()
    }

    default fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[Self::Item; N], array::IntoIter<Self::Item, N>> {
        ByRefDefault::<I>(self.0).next_chunk()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        I::size_hint(self.0)
    }

    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        ByRefDefault::<I>(self.0).advance_by(n)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        ByRefDefault::<I>(self.0).nth(n)
    }

    default fn for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    {
        ByRefDefault::<I>(self.0).for_each(f);
    }

    default fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        ByRefDefault::<I>(self.0).try_fold(init, f)
    }

    default fn try_for_each<F, R>(&mut self, f: F) -> R
    where
        F: FnMut(Self::Item) -> R,
        R: Try<Output = ()>,
    {
        ByRefDefault::<I>(self.0).try_for_each(f)
    }

    default fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        ByRefDefault::<I>(self.0).fold(init, f)
    }

    default fn reduce<F>(self, f: F) -> Option<Self::Item>
    where
        F: FnMut(Self::Item, Self::Item) -> Self::Item,
    {
        ByRefDefault::<I>(self.0).reduce(f)
    }

    default fn try_reduce<R>(
        &mut self,
        f: impl FnMut(Self::Item, Self::Item) -> R,
    ) -> ChangeOutputType<R, Option<R::Output>>
    where
        R: Try<Output = Self::Item, Residual: Residual<Option<Self::Item>>>,
    {
        ByRefDefault::<I>(self.0).try_reduce(f)
    }

    default fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        ByRefDefault::<I>(self.0).all(f)
    }

    default fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        ByRefDefault::<I>(self.0).any(f)
    }

    default fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        ByRefDefault::<I>(self.0).find(predicate)
    }

    default fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        ByRefDefault::<I>(self.0).find_map(f)
    }

    default fn try_find<R>(
        &mut self,
        f: impl FnMut(&Self::Item) -> R,
    ) -> ChangeOutputType<R, Option<Self::Item>>
    where
        R: Try<Output = bool, Residual: Residual<Option<Self::Item>>>,
    {
        ByRefDefault::<I>(self.0).try_find(f)
    }

    default fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        ByRefDefault::<I>(self.0).position(predicate)
    }
}

impl<I: Iterator> Iterator for ByRef<'_, I> {
    fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[Self::Item; N], array::IntoIter<Self::Item, N>> {
        I::next_chunk(self.0)
    }

    fn for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    {
        I::try_for_each(self.0, NeverShortCircuit::wrap_mut_1(f)).0
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        I::try_fold(self.0, init, f)
    }

    fn try_for_each<F, R>(&mut self, f: F) -> R
    where
        F: FnMut(Self::Item) -> R,
        R: Try<Output = ()>,
    {
        I::try_for_each(self.0, f)
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        I::try_fold(self.0, init, NeverShortCircuit::wrap_mut_2(f)).0
    }

    fn reduce<F>(self, f: F) -> Option<Self::Item>
    where
        F: FnMut(Self::Item, Self::Item) -> Self::Item,
    {
        I::try_reduce(self.0, NeverShortCircuit::wrap_mut_2(f)).0
    }

    fn try_reduce<R>(
        &mut self,
        f: impl FnMut(Self::Item, Self::Item) -> R,
    ) -> ChangeOutputType<R, Option<R::Output>>
    where
        R: Try<Output = Self::Item, Residual: Residual<Option<Self::Item>>>,
    {
        I::try_reduce(self.0, f)
    }

    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        I::all(self.0, f)
    }

    fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        I::any(self.0, f)
    }

    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        I::find(self.0, predicate)
    }

    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        I::find_map(self.0, f)
    }

    fn try_find<R>(
        &mut self,
        f: impl FnMut(&Self::Item) -> R,
    ) -> ChangeOutputType<R, Option<Self::Item>>
    where
        R: Try<Output = bool, Residual: Residual<Option<Self::Item>>>,
    {
        I::try_find(self.0, f)
    }

    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        I::position(self.0, predicate)
    }
}

impl<I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for ByRef<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        ByRefDefault::<I>(self.0).next_back()
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        ByRefDefault::<I>(self.0).advance_back_by(n)
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        ByRefDefault::<I>(self.0).nth_back(n)
    }

    default fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        ByRefDefault::<I>(self.0).try_rfold(init, f)
    }

    default fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        ByRefDefault::<I>(self.0).rfold(init, f)
    }

    default fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        ByRefDefault::<I>(self.0).rfind(predicate)
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for ByRef<'_, I> {
    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        I::try_rfold(self.0, init, f)
    }

    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        I::try_rfold(self.0, init, NeverShortCircuit::wrap_mut_2(f)).0
    }

    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        I::rfind(self.0, predicate)
    }
}

impl<I: ExactSizeIterator + ?Sized> ExactSizeIterator for ByRef<'_, I> {
    fn len(&self) -> usize {
        I::len(self.0)
    }

    fn is_empty(&self) -> bool {
        I::is_empty(self.0)
    }
}

impl<I: FusedIterator + ?Sized> FusedIterator for ByRef<'_, I> {}
unsafe impl<I: TrustedLen + ?Sized> TrustedLen for ByRef<'_, I> {}

/// This struct implements `Iterator` the "naive" way without specialization,
/// which is used to access the default implementations of the iterator methods.
struct ByRefDefault<'a, I: ?Sized>(&'a mut I);

// Make sure to add all iterator methods that can be directly forwarded here.
impl<I: Iterator + ?Sized> Iterator for ByRefDefault<'_, I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        I::next(self.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        I::size_hint(self.0)
    }

    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        I::advance_by(self.0, n)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        I::nth(self.0, n)
    }
}

impl<I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for ByRefDefault<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        I::next_back(self.0)
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        I::advance_back_by(self.0, n)
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        I::nth_back(self.0, n)
    }
}

impl<I: ExactSizeIterator + ?Sized> ExactSizeIterator for ByRefDefault<'_, I> {
    fn len(&self) -> usize {
        I::len(self.0)
    }

    fn is_empty(&self) -> bool {
        I::is_empty(self.0)
    }
}

impl<I: FusedIterator + ?Sized> FusedIterator for ByRefDefault<'_, I> {}
unsafe impl<I: TrustedLen + ?Sized> TrustedLen for ByRefDefault<'_, I> {}
