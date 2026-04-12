use core::async_iter::AsyncIterator;
use core::iter::{FusedIterator, TrustedLen};
use core::num::NonZero;
use core::pin::Pin;
use core::slice;
use core::task::{Context, Poll};

use crate::alloc::{Allocator, Global};
#[cfg(not(no_global_oom_handling))]
use crate::borrow::Cow;
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;
use crate::{fmt, vec};

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator + ?Sized, A: Allocator> Iterator for Box<I, A> {
    type Item = I::Item;
    fn next(&mut self) -> Option<I::Item> {
        (**self).next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        (**self).nth(n)
    }
    fn last(self) -> Option<I::Item> {
        BoxIter::last(self)
    }
}

trait BoxIter {
    type Item;
    fn last(self) -> Option<Self::Item>;
}

impl<I: Iterator + ?Sized, A: Allocator> BoxIter for Box<I, A> {
    type Item = I::Item;
    default fn last(self) -> Option<I::Item> {
        #[inline]
        fn some<T>(_: Option<T>, x: T) -> Option<T> {
            Some(x)
        }

        self.fold(None, some)
    }
}

/// Specialization for sized `I`s that uses `I`s implementation of `last()`
/// instead of the default.
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, A: Allocator> BoxIter for Box<I, A> {
    fn last(self) -> Option<I::Item> {
        (*self).last()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator + ?Sized, A: Allocator> DoubleEndedIterator for Box<I, A> {
    fn next_back(&mut self) -> Option<I::Item> {
        (**self).next_back()
    }
    fn nth_back(&mut self, n: usize) -> Option<I::Item> {
        (**self).nth_back(n)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator + ?Sized, A: Allocator> ExactSizeIterator for Box<I, A> {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator + ?Sized, A: Allocator> FusedIterator for Box<I, A> {}

#[unstable(feature = "async_iterator", issue = "79024")]
impl<S: ?Sized + AsyncIterator + Unpin> AsyncIterator for Box<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut **self).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}

/// This implementation is required to make sure that the `Box<[I]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<I, A: Allocator> !Iterator for Box<[I], A> {}

/// This implementation is required to make sure that the `&Box<[I]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<'a, I, A: Allocator> !Iterator for &'a Box<[I], A> {}

/// This implementation is required to make sure that the `&mut Box<[I]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<'a, I, A: Allocator> !Iterator for &'a mut Box<[I], A> {}

// Note: the `#[rustc_skip_during_method_dispatch(boxed_slice)]` on `trait IntoIterator`
// hides this implementation from explicit `.into_iter()` calls on editions < 2024,
// so those calls will still resolve to the slice implementation, by reference.
#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<I, A: Allocator> IntoIterator for Box<[I], A> {
    type IntoIter = vec::IntoIter<I, A>;
    type Item = I;
    fn into_iter(self) -> vec::IntoIter<I, A> {
        self.into_vec().into_iter()
    }
}

#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<'a, I, A: Allocator> IntoIterator for &'a Box<[I], A> {
    type IntoIter = slice::Iter<'a, I>;
    type Item = &'a I;
    fn into_iter(self) -> slice::Iter<'a, I> {
        self.iter()
    }
}

#[stable(feature = "boxed_slice_into_iter", since = "1.80.0")]
impl<'a, I, A: Allocator> IntoIterator for &'a mut Box<[I], A> {
    type IntoIter = slice::IterMut<'a, I>;
    type Item = &'a mut I;
    fn into_iter(self) -> slice::IterMut<'a, I> {
        self.iter_mut()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_slice_from_iter", since = "1.32.0")]
impl<I> FromIterator<I> for Box<[I]> {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<_>>().into_boxed_slice()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl FromIterator<char> for Box<str> {
    fn from_iter<T: IntoIterator<Item = char>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl<'a> FromIterator<&'a char> for Box<str> {
    fn from_iter<T: IntoIterator<Item = &'a char>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl<'a> FromIterator<&'a str> for Box<str> {
    fn from_iter<T: IntoIterator<Item = &'a str>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl FromIterator<String> for Box<str> {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl<A: Allocator> FromIterator<Box<str, A>> for Box<str> {
    fn from_iter<T: IntoIterator<Item = Box<str, A>>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_str_from_iter", since = "1.80.0")]
impl<'a> FromIterator<Cow<'a, str>> for Box<str> {
    fn from_iter<T: IntoIterator<Item = Cow<'a, str>>>(iter: T) -> Self {
        String::from_iter(iter).into_boxed_str()
    }
}

/// This implementation is required to make sure that the `Box<[I; N]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<I, const N: usize, A: Allocator> !Iterator for Box<[I; N], A> {}

/// This implementation is required to make sure that the `&Box<[I; N]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<'a, const N: usize, I, A: Allocator> !Iterator for &'a Box<[I; N], A> {}

/// This implementation is required to make sure that the `&mut Box<[I; N]>: IntoIterator`
/// implementation doesn't overlap with `IntoIterator for T where T: Iterator` blanket.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<'a, const N: usize, I, A: Allocator> !Iterator for &'a mut Box<[I; N], A> {}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<'a, T, const N: usize, A: Allocator> IntoIterator for &'a Box<[T; N], A> {
    type IntoIter = slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<'a, T, const N: usize, A: Allocator> IntoIterator for &'a mut Box<[T; N], A> {
    type IntoIter = slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

/// A by-value `Box<[T; N]>` iterator.
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
#[rustc_insignificant_dtor]
pub struct BoxedArrayIntoIter<T, const N: usize, A: Allocator = Global> {
    // FIXME: make a more efficient implementation (without the need to store capacity)
    inner: vec::IntoIter<T, A>,
}

impl<T, const N: usize, A: Allocator> BoxedArrayIntoIter<T, N, A> {
    /// Returns an immutable slice of all elements that have not been yielded
    /// yet.
    #[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// Returns a mutable slice of all elements that have not been yielded yet.
    #[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> Iterator for BoxedArrayIntoIter<T, N, A> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }

    fn count(self) -> usize {
        self.len()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.inner.advance_by(n)
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> DoubleEndedIterator for BoxedArrayIntoIter<T, N, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, rfold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.rfold(init, rfold)
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.inner.advance_back_by(n)
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> ExactSizeIterator for BoxedArrayIntoIter<T, N, A> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> FusedIterator for BoxedArrayIntoIter<T, N, A> {}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
unsafe impl<T, const N: usize, A: Allocator> TrustedLen for BoxedArrayIntoIter<T, N, A> {}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T: Clone, const N: usize, A: Clone + Allocator> Clone for BoxedArrayIntoIter<T, N, A> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T: fmt::Debug, const N: usize, A: Allocator> fmt::Debug for BoxedArrayIntoIter<T, N, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only print the elements that were not yielded yet: we cannot
        // access the yielded elements anymore.
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

#[stable(feature = "boxed_array_value_iter", since = "CURRENT_RUSTC_VERSION")]
impl<T, const N: usize, A: Allocator> IntoIterator for Box<[T; N], A> {
    type IntoIter = BoxedArrayIntoIter<T, N, A>;
    type Item = T;
    fn into_iter(self) -> BoxedArrayIntoIter<T, N, A> {
        BoxedArrayIntoIter { inner: (self as Box<[T], A>).into_vec().into_iter() }
    }
}
