use core::iter;
use core::mem::MaybeUninit;

use super::Vec;
use crate::alloc::Allocator;

// Specialization trait used for Vec::extend_with
pub(super) trait SpecExtendWith<T> {
    #[track_caller]
    fn spec_extend_with(&mut self, n: usize, value: T);
}

impl<T: Clone, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    #[track_caller]
    default fn spec_extend_with(&mut self, n: usize, value: T) {
        self.extend_trusted(iter::repeat_n(value, n));
    }
}

impl<T: Copy, A: Allocator> SpecExtendWith<T> for Vec<T, A> {
    #[track_caller]
    fn spec_extend_with(&mut self, n: usize, value: T) {
        let len = self.len();
        self.reserve(n);
        let unfilled = self.spare_capacity_mut();

        // SAFETY: the above `reserve` call guarantees `n` to be in bounds.
        let unfilled = unsafe { unfilled.get_unchecked_mut(..n) };
        for elem in unfilled {
            *elem = MaybeUninit::new(value);
        }

        // SAFETY: the elements have been initialized above.
        unsafe { self.set_len(len + n) }
    }
}
