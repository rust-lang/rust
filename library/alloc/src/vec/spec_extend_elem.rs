use core::iter::repeat_n;

use super::Vec;
use crate::alloc::Allocator;

// Specialization trait used for Vec::extend_elem
pub(super) trait SpecExtendElem<T> {
    fn spec_extend_elem(&mut self, n: usize, value: T);
}

impl<T, A: Allocator> SpecExtendElem<T> for Vec<T, A>
where
    T: Clone,
{
    default fn spec_extend_elem(&mut self, n: usize, value: T) {
        self.extend_trusted(repeat_n(value, n))
    }
}

impl<T, A: Allocator> SpecExtendElem<T> for Vec<T, A>
where
    T: Copy,
{
    fn spec_extend_elem(&mut self, n: usize, value: T) {
        self.extend_elem_copy(n, value)
    }
}
