use crate::alloc::Allocator;
use crate::vec;
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
