use core::clone::TrivialClone;
use core::mem::MaybeUninit;
use core::{iter, ptr};

use super::Vec;
use crate::alloc::Allocator;

// Specialization trait used for Vec::extend_with
pub(super) trait SpecExtendWith<T> {
    fn spec_extend_with(&mut self, n: usize, value: T);
}

impl<T: Clone, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    default fn spec_extend_with(&mut self, n: usize, value: T) {
        self.extend_trusted(iter::repeat_n(value, n));
    }
}

impl<T: TrivialClone, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    fn spec_extend_with(&mut self, n: usize, value: T) {
        let len = self.len();
        self.reserve(n);
        let unfilled = self.spare_capacity_mut();

        // SAFETY: the above `reserve` call guarantees `n` to be in bounds.
        let unfilled = unsafe { unfilled.get_unchecked_mut(..n) };

        // SAFETY: because `T` is `TrivialClone`, this is equivalent to calling
        // `T::clone` for every element. Notably, `TrivialClone` also implies
        // that the `clone` implementation will not panic, so we can avoid
        // initialization guards and such.
        unfilled.fill_with(|| MaybeUninit::new(unsafe { ptr::read(&value) }));

        // SAFETY: the elements have been initialized above.
        unsafe { self.set_len(len + n) }
    }
}
