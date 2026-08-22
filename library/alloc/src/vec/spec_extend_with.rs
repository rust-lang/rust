use core::clone::TrivialClone;
use core::{iter, ptr};

use super::Vec;
use crate::alloc::Allocator;

// Specialization trait used for Vec::extend_with
pub(super) trait SpecExtendWith<T> {
    fn spec_extend_with(&mut self, n: usize, value: T);
}

impl<T: Clone, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    #[inline]
    default fn spec_extend_with(&mut self, n: usize, value: T) {
        self.extend_trusted(iter::repeat_n(value, n));
    }
}

impl<T: TrivialClone, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    fn spec_extend_with(&mut self, n: usize, value: T) {
        let len = self.len();
        self.reserve(n);
        let unfilled = self.spare_capacity_mut().as_mut_ptr().cast_init();

        for i in 0..n {
            // SAFETY: `unfilled` is at least as long as `n` thanks
            // to the `reserve` call. Because `T` is `TrivialClone`,
            // this is equivalent to calling `T::clone` for every element.
            unsafe { ptr::write(unfilled.add(i), ptr::read(&value)) };
        }

        // SAFETY: the elements have been initialized above.
        unsafe { self.set_len(len + n) }
    }
}
