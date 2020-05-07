use core::{fmt};
use core::ptr::{self, NonNull};
use core::mem;
use core::iter::FusedIterator;
use core::ops::Try;

use super::{count, RingSlices, VecDeque, wrap_index};
use crate::alloc::{AllocRef, Global};

/// An iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`iter`] method on [`VecDeque`]. See its
/// documentation for more.
///
/// [`iter`]: struct.VecDeque.html#method.iter
/// [`VecDeque`]: struct.VecDeque.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    pub(super) ring: &'a [T],
    pub(super) tail: usize,
    pub(super) head: usize,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        f.debug_tuple("Iter").field(&front).field(&back).finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter { ring: self.ring, tail: self.tail, head: self.head }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail.wrapping_add(1), self.ring.len());
        unsafe { Some(self.ring.get_unchecked(tail)) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = count(self.tail, self.head, self.ring.len());
        (len, Some(len))
    }

    fn fold<Acc, F>(self, mut accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        accum = front.iter().fold(accum, &mut f);
        back.iter().fold(accum, &mut f)
    }

    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        let (mut iter, final_res);
        if self.tail <= self.head {
            // single slice self.ring[self.tail..self.head]
            iter = self.ring[self.tail..self.head].iter();
            final_res = iter.try_fold(init, &mut f);
        } else {
            // two slices: self.ring[self.tail..], self.ring[..self.head]
            let (front, back) = self.ring.split_at(self.tail);
            let mut back_iter = back.iter();
            let res = back_iter.try_fold(init, &mut f);
            let len = self.ring.len();
            self.tail = (self.ring.len() - back_iter.len()) & (len - 1);
            iter = front[..self.head].iter();
            final_res = iter.try_fold(res?, &mut f);
        }
        self.tail = self.head - iter.len();
        final_res
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= count(self.tail, self.head, self.ring.len()) {
            self.tail = self.head;
            None
        } else {
            self.tail = wrap_index(self.tail.wrapping_add(n), self.ring.len());
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.tail == self.head {
            return None;
        }
        self.head = wrap_index(self.head.wrapping_sub(1), self.ring.len());
        unsafe { Some(self.ring.get_unchecked(self.head)) }
    }

    fn rfold<Acc, F>(self, mut accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        accum = back.iter().rfold(accum, &mut f);
        front.iter().rfold(accum, &mut f)
    }

    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        let (mut iter, final_res);
        if self.tail <= self.head {
            // single slice self.ring[self.tail..self.head]
            iter = self.ring[self.tail..self.head].iter();
            final_res = iter.try_rfold(init, &mut f);
        } else {
            // two slices: self.ring[self.tail..], self.ring[..self.head]
            let (front, back) = self.ring.split_at(self.tail);
            let mut front_iter = front[..self.head].iter();
            let res = front_iter.try_rfold(init, &mut f);
            self.head = front_iter.len();
            iter = back.iter();
            final_res = iter.try_rfold(res?, &mut f);
        }
        self.head = self.tail + iter.len();
        final_res
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Iter<'_, T> {
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Iter<'_, T> {}

/// A mutable iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`iter_mut`] method on [`VecDeque`]. See its
/// documentation for more.
///
/// [`iter_mut`]: struct.VecDeque.html#method.iter_mut
/// [`VecDeque`]: struct.VecDeque.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    pub(super) ring: &'a mut [T],
    pub(super) tail: usize,
    pub(super) head: usize,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (front, back) = RingSlices::ring_slices(&*self.ring, self.head, self.tail);
        f.debug_tuple("IterMut").field(&front).field(&back).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail.wrapping_add(1), self.ring.len());

        unsafe {
            let elem = self.ring.get_unchecked_mut(tail);
            Some(&mut *(elem as *mut _))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = count(self.tail, self.head, self.ring.len());
        (len, Some(len))
    }

    fn fold<Acc, F>(self, mut accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        accum = front.iter_mut().fold(accum, &mut f);
        back.iter_mut().fold(accum, &mut f)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= count(self.tail, self.head, self.ring.len()) {
            self.tail = self.head;
            None
        } else {
            self.tail = wrap_index(self.tail.wrapping_add(n), self.ring.len());
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<&'a mut T> {
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.tail == self.head {
            return None;
        }
        self.head = wrap_index(self.head.wrapping_sub(1), self.ring.len());

        unsafe {
            let elem = self.ring.get_unchecked_mut(self.head);
            Some(&mut *(elem as *mut _))
        }
    }

    fn rfold<Acc, F>(self, mut accum: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let (front, back) = RingSlices::ring_slices(self.ring, self.head, self.tail);
        accum = back.iter_mut().rfold(accum, &mut f);
        front.iter_mut().rfold(accum, &mut f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IterMut<'_, T> {
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for IterMut<'_, T> {}

/// An owning iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`into_iter`] method on [`VecDeque`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`into_iter`]: struct.VecDeque.html#method.into_iter
/// [`VecDeque`]: struct.VecDeque.html
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T, A: AllocRef = Global> {
    pub(super) inner: VecDeque<T, A>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug, A: AllocRef> fmt::Debug for IntoIter<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.inner).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: AllocRef> Iterator for IntoIter<T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: AllocRef> DoubleEndedIterator for IntoIter<T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.pop_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, A: AllocRef> ExactSizeIterator for IntoIter<T, A> {
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, A: AllocRef> FusedIterator for IntoIter<T, A> {}



/// A draining iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`drain`] method on [`VecDeque`]. See its
/// documentation for more.
///
/// [`drain`]: struct.VecDeque.html#method.drain
/// [`VecDeque`]: struct.VecDeque.html
#[stable(feature = "drain", since = "1.6.0")]
pub struct Drain<'a, T: 'a, A: AllocRef = Global> {
    pub(super) after_tail: usize,
    pub(super) after_head: usize,
    pub(super) iter: Iter<'a, T>,
    pub(super) deque: NonNull<VecDeque<T, A>>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug, A: AllocRef> fmt::Debug for Drain<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain")
            .field(&self.after_tail)
            .field(&self.after_head)
            .field(&self.iter)
            .finish()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<T: Sync, A: Sync + AllocRef> Sync for Drain<'_, T, A> {}
#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<T: Send, A: Send + AllocRef> Send for Drain<'_, T, A> {}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: AllocRef> Drop for Drain<'_, T, A> {
    fn drop(&mut self) {
        struct DropGuard<'r, 'a, T, A: AllocRef>(&'r mut Drain<'a, T, A>);

        impl<'r, 'a, T, A: AllocRef> Drop for DropGuard<'r, 'a, T, A> {
            fn drop(&mut self) {
                self.0.for_each(drop);

                let source_deque = unsafe { self.0.deque.as_mut() };

                // T = source_deque_tail; H = source_deque_head; t = drain_tail; h = drain_head
                //
                //        T   t   h   H
                // [. . . o o x x o o . . .]
                //
                let orig_tail = source_deque.tail;
                let drain_tail = source_deque.head;
                let drain_head = self.0.after_tail;
                let orig_head = self.0.after_head;

                let tail_len = count(orig_tail, drain_tail, source_deque.cap());
                let head_len = count(drain_head, orig_head, source_deque.cap());

                // Restore the original head value
                source_deque.head = orig_head;

                match (tail_len, head_len) {
                    (0, 0) => {
                        source_deque.head = 0;
                        source_deque.tail = 0;
                    }
                    (0, _) => {
                        source_deque.tail = drain_head;
                    }
                    (_, 0) => {
                        source_deque.head = drain_tail;
                    }
                    _ => unsafe {
                        if tail_len <= head_len {
                            source_deque.tail = source_deque.wrap_sub(drain_head, tail_len);
                            source_deque.wrap_copy(source_deque.tail, orig_tail, tail_len);
                        } else {
                            source_deque.head = source_deque.wrap_add(drain_tail, head_len);
                            source_deque.wrap_copy(drain_tail, drain_head, head_len);
                        }
                    },
                }
            }
        }

        while let Some(item) = self.next() {
            let guard = DropGuard(self);
            drop(item);
            mem::forget(guard);
        }

        DropGuard(self);
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: AllocRef> Iterator for Drain<'_, T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|elt| unsafe { ptr::read(elt) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: AllocRef> DoubleEndedIterator for Drain<'_, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|elt| unsafe { ptr::read(elt) })
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: AllocRef> ExactSizeIterator for Drain<'_, T, A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, A: AllocRef> FusedIterator for Drain<'_, T, A> {}
