use core::fmt;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::ptr::{self, NonNull};

use crate::alloc::{Allocator, Global};

use super::{count, wrap_index, VecDeque};

/// A draining iterator over the elements of a `VecDeque`.
///
/// This `struct` is created by the [`drain`] method on [`VecDeque`]. See its
/// documentation for more.
///
/// [`drain`]: VecDeque::drain
#[stable(feature = "drain", since = "1.6.0")]
pub struct Drain<
    'a,
    T: 'a,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    after_tail: usize,
    after_head: usize,
    ring: NonNull<[T]>,
    tail: usize,
    head: usize,
    deque: NonNull<VecDeque<T, A>>,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator> Drain<'a, T, A> {
    pub(super) unsafe fn new(
        after_tail: usize,
        after_head: usize,
        ring: &'a [MaybeUninit<T>],
        tail: usize,
        head: usize,
        deque: NonNull<VecDeque<T, A>>,
    ) -> Self {
        let ring = unsafe { NonNull::new_unchecked(ring as *const [MaybeUninit<T>] as *mut _) };
        Drain { after_tail, after_head, ring, tail, head, deque, _phantom: PhantomData }
    }
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug, A: Allocator> fmt::Debug for Drain<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Drain")
            .field(&self.after_tail)
            .field(&self.after_head)
            .field(&self.ring)
            .field(&self.tail)
            .field(&self.head)
            .finish()
    }
}

#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<T: Sync, A: Allocator + Sync> Sync for Drain<'_, T, A> {}
#[stable(feature = "drain", since = "1.6.0")]
unsafe impl<T: Send, A: Allocator + Send> Send for Drain<'_, T, A> {}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: Allocator> Drop for Drain<'_, T, A> {
    fn drop(&mut self) {
        struct DropGuard<'r, 'a, T, A: Allocator>(&'r mut Drain<'a, T, A>);

        impl<'r, 'a, T, A: Allocator> Drop for DropGuard<'r, 'a, T, A> {
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
impl<T, A: Allocator> Iterator for Drain<'_, T, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail.wrapping_add(1), self.ring.len());
        // Safety:
        // - `self.tail` in a ring buffer is always a valid index.
        // - `self.head` and `self.tail` equality is checked above.
        unsafe { Some(ptr::read(self.ring.as_ptr().get_unchecked_mut(tail))) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = count(self.tail, self.head, self.ring.len());
        (len, Some(len))
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: Allocator> DoubleEndedIterator for Drain<'_, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.tail == self.head {
            return None;
        }
        self.head = wrap_index(self.head.wrapping_sub(1), self.ring.len());
        // Safety:
        // - `self.head` in a ring buffer is always a valid index.
        // - `self.head` and `self.tail` equality is checked above.
        unsafe { Some(ptr::read(self.ring.as_ptr().get_unchecked_mut(self.head))) }
    }
}

#[stable(feature = "drain", since = "1.6.0")]
impl<T, A: Allocator> ExactSizeIterator for Drain<'_, T, A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, A: Allocator> FusedIterator for Drain<'_, T, A> {}
