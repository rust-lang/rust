use core::clone::TrivialClone;
use core::iter::{Cloned, TrustedLen};
use core::slice;

use super::{IntoIter, Vec};
use crate::alloc::Allocator;

// Specialization trait used for Vec::extend
pub(super) trait SpecExtend<I> {
    fn spec_extend(&mut self, iter: I);
}

impl<T, I, A: Allocator> SpecExtend<I> for Vec<T, A>
where
    I: Iterator<Item = T>,
{
    default fn spec_extend(&mut self, iter: I) {
        self.extend_desugared(iter)
    }
}

impl<T, I, A: Allocator> SpecExtend<I> for Vec<T, A>
where
    I: TrustedLen<Item = T>,
{
    default fn spec_extend(&mut self, iterator: I) {
        self.extend_trusted(iterator)
    }
}

impl<T, A1: Allocator, A2: Allocator> SpecExtend<IntoIter<T, A2>> for Vec<T, A1> {
    fn spec_extend(&mut self, mut iterator: IntoIter<T, A2>) {
        unsafe {
            self.append_elements(iterator.as_slice() as _);
        }
        iterator.forget_remaining_elements();
    }
}

impl<'a, T: 'a, I, A: Allocator> SpecExtend<Cloned<slice::Iter<'a, T>>> for Vec<T, A>
where
    T: TrivialClone,
{
    fn spec_extend(&mut self, mut iterator: Cloned<slice::Iter<'a, T>>) {
        let inner: &mut slice::Iter<'a, T> = unsafe { iterator.as_inner() };
        let slice = iterator.as_slice();
        unsafe { self.append_elements(slice); }
    }
}
