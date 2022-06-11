use crate::alloc::Allocator;
use crate::vec;
use core::iter::{ByRefSized, TrustedLen};
use core::slice;

use super::VecDeque;

// Specialization trait used for VecDeque::extend
pub(super) trait SpecExtend<T, I> {
    fn spec_extend(&mut self, iter: I);
}

impl<T, I, A: Allocator> SpecExtend<T, I> for VecDeque<T, A>
where
    I: Iterator<Item = T>,
{
    default fn spec_extend(&mut self, mut iter: I) {
        // This function should be the moral equivalent of:
        //
        //      for item in iter {
        //          self.push_back(item);
        //      }
        while let Some(element) = iter.next() {
            if self.len() == self.capacity() {
                let (lower, _) = iter.size_hint();
                self.reserve(lower.saturating_add(1));
            }

            let head = self.head;
            self.head = self.wrap_add(self.head, 1);
            unsafe {
                self.buffer_write(head, element);
            }
        }
    }
}

impl<T, I, A: Allocator> SpecExtend<T, I> for VecDeque<T, A>
where
    I: TrustedLen<Item = T>,
{
    default fn spec_extend(&mut self, mut iter: I) {
        // This is the case for a TrustedLen iterator.
        let (low, high) = iter.size_hint();
        if let Some(additional) = high {
            debug_assert_eq!(
                low,
                additional,
                "TrustedLen iterator's size hint is not exact: {:?}",
                (low, high)
            );
            self.reserve(additional);

            struct WrapAddOnDrop<'a, T, A: Allocator> {
                vec_deque: &'a mut VecDeque<T, A>,
                written: usize,
            }

            impl<'a, T, A: Allocator> Drop for WrapAddOnDrop<'a, T, A> {
                fn drop(&mut self) {
                    self.vec_deque.head =
                        self.vec_deque.wrap_add(self.vec_deque.head, self.written);
                }
            }

            let mut wrapper = WrapAddOnDrop { vec_deque: self, written: 0 };

            let head_room = wrapper.vec_deque.cap() - wrapper.vec_deque.head;
            unsafe {
                wrapper.vec_deque.write_iter(
                    wrapper.vec_deque.head,
                    ByRefSized(&mut iter).take(head_room),
                    &mut wrapper.written,
                );

                if additional > head_room {
                    wrapper.vec_deque.write_iter(0, iter, &mut wrapper.written);
                }
            }

            debug_assert_eq!(
                additional, wrapper.written,
                "The number of items written to VecDeque doesn't match the TrustedLen size hint"
            );
        } else {
            // Per TrustedLen contract a `None` upper bound means that the iterator length
            // truly exceeds usize::MAX, which would eventually lead to a capacity overflow anyway.
            // Since the other branch already panics eagerly (via `reserve()`) we do the same here.
            // This avoids additional codegen for a fallback code path which would eventually
            // panic anyway.
            panic!("capacity overflow");
        }
    }
}

impl<T, A: Allocator> SpecExtend<T, vec::IntoIter<T>> for VecDeque<T, A> {
    fn spec_extend(&mut self, mut iterator: vec::IntoIter<T>) {
        let slice = iterator.as_slice();
        self.reserve(slice.len());

        unsafe {
            self.copy_slice(self.head, slice);
            self.head = self.wrap_add(self.head, slice.len());
        }
        iterator.forget_remaining_elements();
    }
}

impl<'a, T: 'a, I, A: Allocator> SpecExtend<&'a T, I> for VecDeque<T, A>
where
    I: Iterator<Item = &'a T>,
    T: Copy,
{
    default fn spec_extend(&mut self, iterator: I) {
        self.spec_extend(iterator.copied())
    }
}

impl<'a, T: 'a, A: Allocator> SpecExtend<&'a T, slice::Iter<'a, T>> for VecDeque<T, A>
where
    T: Copy,
{
    fn spec_extend(&mut self, iterator: slice::Iter<'a, T>) {
        let slice = iterator.as_slice();
        self.reserve(slice.len());

        unsafe {
            self.copy_slice(self.head, slice);
            self.head = self.wrap_add(self.head, slice.len());
        }
    }
}
