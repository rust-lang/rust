use core::async_iter::AsyncIterator;
use core::iter::FusedIterator;
use core::pin::Pin;
use core::slice;
use core::task::{Context, Poll};

use crate::alloc::Allocator;
#[cfg(not(no_global_oom_handling))]
use crate::borrow::Cow;
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
use crate::vec;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

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
